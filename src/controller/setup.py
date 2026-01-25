from setuptools import setup
import os
from glob import glob

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 런치 파일 설치 경로 재확인
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='KMC Controller Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cav1 = controller.drive1:main',
            'cav2 = controller.drive2:main',
            'cav3 = controller.drive3:main',
            'cav4 = controller.drive4:main',
            'drive_basic = controller.drive_basic:main', # <-- 추가됨
            'smart_infra = controller.smartinfra:main',
        ],
    },
)