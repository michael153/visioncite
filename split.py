import argparse
import math
import shutil
import os


def main():
    """Main program.

    Returns:
        Zero on succesful termination, non-zero otherwise.

    Example:
        $ split.py --percent 0.85 images xtrain xvalid
    """
    parser = argparse.ArgumentParser(description='Splits directory into two new directories')
    parser.add_argument('directory',
                        metavar='directory',
                        help='path to a directory')
    parser.add_argument('new_dir1',
                        help='desired output for the first new directory')
    parser.add_argument('new_dir2',
                        help='desired output for the second new directory')
    parser.add_argument('--percent',
                        type=float,
                        dest='percent',
                        default=0.8,
                        help='percent of files split into the first new directory (default: .8)')
    args = parser.parse_args()

    num_files = len(os.listdir(args.directory))
    split_size = math.floor(num_files * args.percent)
    split(args.directory, args.new_dir1, args.new_dir2, split_size)

    return 0


def make_output_directories(new_dir1, new_dir2):
    if not os.path.exists(new_dir1):
        os.makedirs(new_dir1)
    if not os.path.exists(new_dir2):
        os.makedirs(new_dir2)


def split(directory, new_dir1, new_dir2, size):
    make_output_directories(new_dir1, new_dir2)

    filenames = [filename for filename in os.listdir(directory)]
    for i, filename in enumerate(filenames):
        if i <= size:
            shutil.copyfile(directory + "/" + filename, new_dir1 + "/" + filename)
        else:
            shutil.copyfile(directory + "/"+ filename, new_dir2 + "/" + filename)

        percent = "{0:.0%}".format(i / len(filenames))
        print("Copying files | %s | %s" % (percent, filename), end="\r")


if __name__ == "__main__":
    main()
