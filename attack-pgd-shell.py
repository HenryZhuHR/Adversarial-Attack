import os
BATCH = 12



for EPSILON in (1, 4, 16, 64):
    for ALPHA in (1, 4, 16, 64):
        for NUM_STEPS in (40, 20, 10, 5):
            cmd = ''
            cmd += 'python3 attack-pgd.py'
            cmd += ' --batch_size  %d' % BATCH
            cmd += ' --num_steps   %d' % NUM_STEPS
            cmd += ' --epsilon     %d' % EPSILON
            cmd += ' --alpha       %d' % ALPHA
            cmd += ' --num_worker  %d' % (0)
            cmd += ' --device      %s' % 'cuda:0'            
            cmd += ' --pretrained  %s' % ('checkpoints/noAttack.pt')
            os.system(cmd)
