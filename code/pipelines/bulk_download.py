import os
from LabUtils.addloglevels import sethandlers
from LabUtils.Scripts.shell_commands_execute import ShellCommandsExecute

from LabQueue.qp import qp # fakeqp as qp #
from LabData.DataAnalyses.UKBB_10k.UKBB_survival import *


def run_download(lineno):
    os.chdir("/net/mraid20/export/jasmine/UKBB_DATA")
  #  os.chdir("/net/mraid08/export/jafar/UKBioBank/Data/davidpel_download/")
    cmd = f'./ukbfetch -bukb673549.bulk -ak28784r673549.key -s{1 + (1000 * lineno)} -m1000'
  #  cmd = f'./ukbfetch -bukb673055.bulk -ak28784r673055.key -s{1+(1000*lineno)} -m1000'
    ShellCommandsExecute().run(cmd=cmd, cmd_name=f'download {lineno}')
    return lineno

def main():

    bulk_file = "/net/mraid20/export/jasmine/UKBB_DATA/ukb673549.bulk"
    #bulk_file = "/net/mraid08/export/jafar/UKBioBank/Data/davidpel_download/ukb673055.bulk"
    num_lines = 0
    with open(bulk_file,'r') as f:
        for line in f:
            num_lines += 1
    num_threads = (num_lines // 1000) + 1
    print(num_threads)

    old_cwd = os.getcwd()
    os.chdir(OUTPATH)

    sethandlers()

    with qp(jobname='bulk_download',  _trds_def=1, max_r=20,  _mem_def='2G') as q:
        q.startpermanentrun()
        res = []
        for i in range(num_threads):
            res.append(q.method(run_download, (i,)))
        q.waitforresults(res)

    os.chdir(old_cwd)




if __name__ == "__main__":
    main()