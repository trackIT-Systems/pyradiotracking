import logging
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from configparser import ConfigParser
from typing import Any, Dict, List, Optional, Sequence, Text, TextIO, Tuple


# argparse names the group of optional arguments "optional arguments" before Python 3.10 and "options" since.
# Config files written with either name are read, regardless of the running Python version.
SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "options": ("optional arguments",),
    "optional arguments": ("options",),
}


class ArgConfParser(ArgumentParser):
    """
    An argparse.ArgumentParser subclass that reads a config file and updates the namespace accordingly.

    Parameters
    ----------
    config_dest: str
        The name of the config file to be read.
    """

    def __init__(self, *args, config_dest=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_dest = config_dest

    def parse_known_args(
        self, args: Optional[Sequence[Text]] = None, namespace: Optional[Namespace] = None
    ) -> Tuple[Namespace, List[str]]:
        """
        Initialize defaults according to the parser, update with config file and then parse the arguments.

        Parameters
        ----------
        args: typing.Optional[typing.Sequence[str]]
            The list of arguments to parse.
        namespace: typing.Optional[argparse.Namespace]
            An optional initial namespace.

        Returns
        -------
        argparse.Namespace
            The parsed namespace.
        typing.List[str]
            The remaining arguments.
        """
        # create namespace filled with default values
        namespace, _ = super().parse_known_args(args=[], namespace=namespace)

        # read config parameter to separate namespace
        config_namespace, _ = super().parse_known_args(args=args)

        # read config file if specified and update namespace
        if self.config_dest in config_namespace.__dict__:
            config = self.read_config(config_namespace.__dict__[self.config_dest])
            namespace.__dict__.update(config)

        # parse args and update namespace
        namespace, unparsed = super().parse_known_args(args=args, namespace=namespace)

        return (namespace, unparsed)

    def read_config(self, path: str) -> Dict[str, Any]:
        """
        Read a config file and update the namespace accordingly.

        Parameters
        ----------
        path: str
            The path to the config file.

        Returns
        -------
        typing.Dict[str, typing.Any]
            The configuration read from the file.
        """
        # values are python literals, '%' must not be interpreted
        config = ConfigParser(interpolation=None)
        config.read(path)

        conf_dict = {}

        for group in self._action_groups:
            # skip untitled groups
            if not isinstance(group.title, str):
                continue

            # sections to read, aliases first so that the group's own title wins if both are present
            sections = [name for name in (*SECTION_ALIASES.get(group.title, ()), group.title) if name in config]

            # iterate actions and extract values
            for action in group._group_actions:
                for section in sections:
                    if action.dest in config[section]:
                        conf_dict[action.dest] = literal_eval(config[section][action.dest])

        return conf_dict

    def write_config(self, args: Namespace, file: TextIO, help: bool = False):
        """
        Writes the current namespace to a config file.

        Parameters
        ----------
        args: argparse.Namespace
            The namespace to write.
        file: typing.TextIO
            The file to write to.
        help: bool
            Whether to write the help text.
        """
        # written by hand: ConfigParser lowercases or rejects the help texts when used as comment "keys"
        sections = []

        for group in self._action_groups:
            # skip unnamed groups
            if not isinstance(group.title, str):
                continue

            # iterate actions and extract parameters from args
            lines = []
            for action in group._group_actions:
                if action.dest in args.__dict__:
                    if help:
                        lines.append(f"# {action.help}")
                    lines.append(f"{action.dest} = {args.__dict__[action.dest]!r}")

            # skip empty groups
            if lines:
                sections.append("\n".join([f"[{group.title}]", *lines]))

        file.write("\n\n".join(sections) + "\n\n")

