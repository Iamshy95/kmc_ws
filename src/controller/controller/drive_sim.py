# [상단 설정 구역 - SIM]
from config import PARAMS, LOG_HEADERS, ENV_CONFIG
from geometry_msgs.msg import Accel as CMD_MSG_TYPE # 시뮬은 Accel

MODE = "SIM"
CFG = ENV_CONFIG[MODE]
DT = CFG["dt"]
# 시뮬은 ID를 코드에 박거나 런칭 시 포맷팅
CAR_ID = 1 
TOPIC_POSE = CFG["topic_pose"].format(id=CAR_ID)
TOPIC_CMD = CFG["topic_cmd"].format(id=CAR_ID)
USE_MA = CFG["use_ma_input"]