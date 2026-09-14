from setuptools import find_packages, setup

package_name = 'publish_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='murderer0107@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_detection = publish_package.yolo_detection:main',
            'test2._subscriber추가 = publish_package.test2._subscriber추가:main',
            'yolo_detection_turtlebot = publish_package.yolo_detection_turtlebot:main',
            'yolo_detection_tomato = publish_package.yolo_detection_tomato:main',
            'cctv = publish_package.cctv_camera_publisher:main',
        ],
    },
)
