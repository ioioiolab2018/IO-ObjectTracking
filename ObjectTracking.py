import sys
import algorithms


def main():
    print_help()

    key = input()
    key = key.rstrip('\n')

    if key == "1":
        algorithms.color_run()
    elif key == "2":
        algorithms.csrt_run()
    elif key == "3":
        algorithms.face_run()
    elif key == "q":
        sys.exit(0)


def print_help():
    print("Welcome user in the OBJECT TRACKING program! You have several options:")
    print("\t 1 - color algorithm")
    print("\t 2 - csrt algorithm")
    print("\t 3 - face tracking algorithm (Haar algorithm + csrt algorithm)")
    print("\t q - quit program")
    print("Select option to continue:")


if __name__ == '__main__':
    main()
