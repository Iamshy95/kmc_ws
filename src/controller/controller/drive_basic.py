# [상단 설정 구역 - REAL]
from config import PARAMS, LOG_HEADERS, ENV_CONFIG
from geometry_msgs.msg import Twist as CMD_MSG_TYPE # 실차는 Twist

MODE = "REAL"
CFG = ENV_CONFIG[MODE]
DT = CFG["dt"]
TOPIC_POSE = CFG["topic_pose"]
TOPIC_CMD = CFG["topic_cmd"]
USE_MA = CFG["use_ma_input"]