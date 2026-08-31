"""Run a command fully detached: double-fork + setsid + no controlling terminal.

WHY. Long LAMMPS campaigns launched from this session get SIGSTOP'd (process
state T) whenever the foreground call that spawned them returns -- they sit
there accruing no CPU, silently, looking exactly like a slow run. `nohup`,
`</dev/null` and the harness's own background mode all failed to prevent it.
A double fork with os.setsid() severs the controlling terminal and the process
group, which does.

Usage:  python3 daemonize.py <logfile> <cmd> [args ...]
"""
import os
import sys

log, cmd = sys.argv[1], sys.argv[2:]
if os.fork():
    os._exit(0)
os.setsid()
if os.fork():
    os._exit(0)
fd = os.open(os.devnull, os.O_RDONLY)
os.dup2(fd, 0)
out = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(out, 1)
os.dup2(out, 2)
os.execvp(cmd[0], cmd)
