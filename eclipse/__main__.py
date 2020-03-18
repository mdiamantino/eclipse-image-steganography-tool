"""Eclipse

Usage:
  __main__.py --interactive
  __main__.py hide --image <image-path> --message <message-txt> --code <seed> --output <stego-path>
  __main__.py extract --image <image-path> --code <seed>
  __main__.py (-h | --help)
  __main__.py --version

Options:
  --interactive                                 Enables interactive CLI.
  -i <image-path>, --image <image-path>         Path of the original | stegoimage.
  -m <image-path>, --message <message-txt>      Message to hide.
  -c <seed>, --code <seed>                      Repartition code used to retrieve the message
  -o <stego-path>, --output <stego-path>        Path of the output stegoimage.
  -h, --help    Show this screen.
  --version     Show version.
"""
import getpass

from eclipse import main_app
from pyfiglet import Figlet
from docopt import docopt


def main():
    f = Figlet(font='isometric3')
    print(f.renderText('HIDE'))
    arguments = docopt(__doc__, version='DEMO 1.0')
    if arguments['--interactive']:
        main_app.interactive_cli()
    else:
        image_path = arguments['--image']
        repartition_code = int(arguments['--code'])
        password = getpass.getpass(prompt='Password: ', stream=None)
        if arguments['hide']:
            message_to_hide = arguments['--message']
            output_image_path = arguments['--output']
            main_app.embed(original_image_path=image_path,
                           stego_image_output_path=output_image_path,
                           message=message_to_hide,
                           password=password,
                           chosen_seed=repartition_code)
        elif arguments['extract']:
            retrieved_message = main_app.extract(stego_img_path=image_path,
                                                 password=password,
                                                 chosen_seed=repartition_code)


if __name__ == "__main__":
    # main_app.command_line_interface_main()
    arguments = docopt(__doc__, version='DEMO 1.0')
    print(arguments)
