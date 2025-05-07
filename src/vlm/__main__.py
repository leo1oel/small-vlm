import os
import sys

from .vlm import main

if __name__ == "__main__":
    local_rank = None
    i = 0
    while i < len(sys.argv):
        if sys.argv[i].startswith("--local_rank="):
            local_rank = int(sys.argv[i].split("=")[1])
            sys.argv.pop(i)
        else:
            i += 1

    if local_rank is not None:
        os.environ["LOCAL_RANK"] = str(local_rank)
    main()
