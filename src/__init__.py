import os
import pathlib
import sys


def getResource(filename: pathlib.Path) -> pathlib.Path:
    """
    A useful function to identify the absolute filepath of a resource if using PyInstaller  --add-data.

    If you are not using PyInstaller, do not leverage this method.

    :param filename: A filename to identify the absolute path of
    :type filename: pathlib.Path
    :return: A filepath to a resource compatible with PyInstaller --add-data
    :rtype: pathlib.Path
    """  # noqa: E501
    base: str = getattr(
        sys,
        "_MEIPASS",
        os.path.dirname(os.path.abspath(__file__)),
    )

    return pathlib.Path(base, filename)


def getVersion() -> str:
    """
    Wrapper around getResource to identify the location of a _version file

    :return: The contents of ./_version
    :rtype: str
    """  # noqa: E501
    versionFile: pathlib.Path = getResource(filename="_version")

    if os.path.exists(path=versionFile):
        return open(file=versionFile, mode="r").read().strip()
    else:
        return "?.?.?"
