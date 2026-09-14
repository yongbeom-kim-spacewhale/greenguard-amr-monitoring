import os
from re import L
import sqlite3
from tkinter import EXCEPTION

# =================================================================
# 📌 프로그램 & 클래스 요약: GreenGuard 관제 데이터베이스 관리 엔진
# =================================================================
# 본 클래스는 스마트팜 통합 관제 시스템의 사용자 계정 정보와 터틀봇 감지 로그를
# 영구 저장하고 관리하는 SQLite3 전용 래퍼(Wrapper) 데이터베이스 엔진임.
# 파이썬의 'with' 문(Context Manager) 프로토콜을 구현하여, 웹 서버 내부에서
# 커넥션 누수(Connection Leak) 및 데이터베이스 락(Lock) 현상을 원천 차단함.
# 시스템 가동 시 초기 테이블 셋업, 사용자 검증, 30일 경과 로그 다이어트 기능을 처리함.
# =================================================================

class GreenGuardDB:
    def __init__(self, db_name='green_guard.db'):
        # 실행시 안전하게 db_name만 연결
        self.db_name = db_name
        # 객체(Flask 서버 실행 시) 테이블 없으면 만들기. 서버 껐다 켤때만 실행됨.
        self.create_tables()
        self.seed_admin()

    def __enter__(self):
        # 'with db as active_db:' 문이 시작될 때 파이썬 엔진에 의해 자동 실행되는 진입 함수
        # with 문 실행시 자동으로 연결 생성
        self.conn = sqlite3.connect(self.db_name)

        # 쿼리 결과 레코드를 단순 튜플()이 아닌 딕셔너리 형태 {'id': 1, 'image_name': '...'}로 꺼내오도록 팩토리 세팅
        self.conn.row_factory = sqlite3.Row
        return self # with 문 우측의 active_db 변수로 이 객체 인스턴스 자체를 넘겨줌

    def __exit__(self, exc_type, exc_value, traceback):
        # 'with' 블록이 정상적으로 끝나거나, 내부에서 에러가 터져 튕겨 나갈 때 100% 자동 호출되는 안전장치 함수
        # with 문 종료시 자동으로 연결 종료(정상 종료가 되었든 에러로 종료가 되었든 간에 무조건 db와의 접속 차단으로 db 안전하게 보관)
        self.conn.close()

    def create_tables(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # 계정 테이블 생성
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users (
                           username TEXT PRIMARY KEY UNIQUE,
                           password TEXT NOT NULL
                       )
        ''')

        # 감지 테이블 생성
        cursor.execute('''
                        CREATE TABLE IF NOT EXISTS detection_table (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            image_name TEXT NOT NULL,
                            log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            pos_x REAL DEFAULT 0.0,
                            pos_y REAL DEFAULT 0.0
                        )
        ''')    # AUTOINCREMENT 사용시 번호 저장하기위해 비밀 테이블 하나 만들기 때문에 sqlite_sequence 테이블 자동으로 생김

        conn.commit()
        conn.close()

    def seed_admin(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # 관리자 계정이 없다면 생성
            cursor.execute('INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)', ('admin', '1234'))
            conn.commit()
        
    def verify_user(self, username, password):
        # 로그인 검증용 로직
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? and password = ?', (username, password))
        return cursor.fetchone() is not None

    def insert_detection_log(self, image_name, log_time, pos_x, pos_y):
        # 터틀봇 이벤트 발생 시 로그 적재
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO detection_table (image_name, log_time, pos_x, pos_y) VALUES(?, ?, ?, ?)', (image_name, log_time, pos_x, pos_y))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_logs(self):
        # 탭 2 화면용 전체 로그 최신순 조회
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM detection_table ORDER BY id DESC')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_logs_by_date(self, search_date):
        # 사용자가 입력한 특정 날짜(예: '2026-06-26')의 로그만 조회하는 함수
        cursor = self.conn.cursor()
        # LIKE '2026-06-26%' 구문을 사용하여 해당 날짜에 찍힌 모든 시분초 로그를 긁어옵니다.
        query = "SELECT id, image_name, log_time FROM detection_table WHERE log_time LIKE ? ORDER BY id DESC"
        cursor.execute(query, (f"{search_date}%",))
        rows = cursor.fetchall()
        
        # 기존 get_all_logs와 동일하게 자바스크립트가 읽기 편한 딕셔너리 리스트로 변환
        return [dict(row) for row in rows]
    
    def delete_after_month(self):
        cursor = self.conn.cursor()
        cursor.execute("""
                       DELETE FROM detection_table
                       WHERE log_time < datetime('now', '-30 days')
                       """)
        self.conn.commit()    
    
    def delete_logs_by_count(self, max_count=500, img_dir_path=None):
        """ 
        데이터 개수가 max_count(500개)를 초과하면 오래된 순으로 잘라내는 함수
        물리 파일 삭제를 위해 이미지 저장 경로(image_dir_path)를 매개변수로 받음.
        """
        cursor = self.conn.cursor()

        try:
            # 1. 현재 저장된 전체 데이터 개수 확인
            cursor.execute("SELECT COUNT(*) FROM detection_table")
            current_count = cursor.fetchone()[0]

            # 500개 이하이면 지울 필요가 없으므로 조용히 종료
            if current_count <= max_count:
                return

            # 2. 초과된 개수 계산 (예: 505개면 5개 삭제 필요)
            overflow_count = current_count - max_count


            # 3. 삭제할 예정인 가장 오래된 데이터의 '이미지 파일명' 목록 추출 (id 오름차순 = 옛날 데이터)
            cursor.execute(
                "SELECT id, image_name FROM detection_table ORDER BY id ASC LIMIT ?",
                (overflow_count,)
            )
            target_rows = [dict(row) for row in cursor.fetchall()]
            
            # 물리 파일 삭제용 이름 추출
            old_images = [row['image_name'] for row in target_rows]
            # 정확한 타깃 매핑을 위한 ID 추출
            target_ids = [row['id'] for row in target_rows]

            if not target_ids:
                return
            
            placeholders = ','.join('?' for _ in target_ids)
            cursor.execute(
                f"DELETE FROM detection_table WHERE id IN ({placeholders})",
                target_ids
            )
            self.conn.commit()

            if img_dir_path and os.path.exists(img_dir_path):
                for img_name in old_images:
                    file_path = os.path.join(img_dir_path, img_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
            print(f"[DB 관리] 데이터 상한선 초과로 인해 오래된 로그 및 파일 {overflow_count}개 청소 완료.")

        except Exception as e:
            self.conn.rollback()
            print(f"[DB 에러] 500 개수 기준 데이터 삭제 처리 실패: {e}")


if __name__ == '__main__':
    print('=== 스마트팜 DB 초기화 및 생성 테스트 시작 ===')

    db = GreenGuardDB('green_guard.db')

    with db as active_db:
        active_db.seed_admin()

    print('테스트 완료!')