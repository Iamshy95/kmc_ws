import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 패키지 이름 설정 (사용자의 패키지명에 맞게 수정하세요. 여기선 'controller'로 가정)
    pkg_name = 'controller'

    return LaunchDescription([
        # 1. 스마트 인프라 노드
        Node(
            package=pkg_name,
            executable='smartinfra_sim',
            name='smart_infra_manager',
            output='screen'
        ),

        # 2. CAV 1번 주행 노드
        Node(
            package=pkg_name,
            executable='drive1_sim',
            name='cav_01_driver',
            output='screen'
        ),

        # 3. CAV 2번 주행 노드
        Node(
            package=pkg_name,
            executable='drive2_sim',
            name='cav_02_driver',
            output='screen'
        ),

        # 4. CAV 3번 주행 노드
        Node(
            package=pkg_name,
            executable='drive3_sim',
            name='cav_03_driver',
            output='screen'
        ),

        # 5. CAV 4번 주행 노드
        Node(
            package=pkg_name,
            executable='drive4_sim',
            name='cav_04_driver',
            output='screen'
        ),
    ])