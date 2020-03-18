from eclipse import main_app
import click


@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.option('--name', prompt='Your name', help='The person to greet.')


if __name__ == "__main__":
    main_app.command_line_interface_main()
