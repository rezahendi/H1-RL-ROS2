"""Let `pytest` run from the repository root.

The ROS 2 package lives in `src/h1_rl`, so that directory has to be on `sys.path`
for `import h1_rl` to work without building or installing the workspace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "h1_rl"))
