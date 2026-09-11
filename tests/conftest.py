import sys
from pathlib import Path


# Lấy thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thêm project root vào Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))