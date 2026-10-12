import os, sys, paramiko
c=paramiko.SSHClient(); c.load_system_host_keys()
c.connect('192.168.20.126',username='saeed',password=os.environ['SSH_PASSWORD'],timeout=15)
if sys.argv[1:2]==['put']:
    s=c.open_sftp(); s.put(sys.argv[2],sys.argv[3]); s.close()
elif sys.argv[1:2]==['get']:
    s=c.open_sftp(); s.get(sys.argv[2],sys.argv[3]); s.close()
else:
    cmd=sys.stdin.read() if len(sys.argv)==1 else sys.argv[1]
    _,out,_=c.exec_command(cmd+' 2>&1'); out.channel.set_combine_stderr(True)
    for line in out: print(line,end='',flush=True)
    rc=out.channel.recv_exit_status();c.close();sys.exit(rc)
c.close()
