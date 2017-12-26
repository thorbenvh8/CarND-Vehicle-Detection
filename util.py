# Print iterations progress - Found https://stackoverflow.com/a/34325723
def printProgressBar (iteration, total, name, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%21s - %s |%s| %5s%% %s' % (name, prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

from random import shuffle
# Shuffle list - Found https://stackoverflow.com/a/47767609
def shuffle_list(*ls):
  l =list(zip(*ls))
  shuffle(l)
  return zip(*l)
