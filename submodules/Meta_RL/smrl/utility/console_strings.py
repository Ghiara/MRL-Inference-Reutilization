verbose = True

def print_to_terminal(text: str):
    global verbose
    if verbose:
        print(text)



# Reference: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
# Reference: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797

from typing import Tuple

# Control characters
ESC = '\033'    # escape code, starts a command
RESET = ESC+'[0m'   # reset style


# Font
def reset_after_string(s: str):
    return s + RESET

def bold(s: str):
    return ESC + "[1m" + s + ESC + "[22m"

def dim(s: str):
    return ESC + "[2m" + s + ESC + "[22m"

def underline(s: str):
    return ESC + "[4m" + s + ESC + "[24m"

def italic(s: str):
    return ESC + "[3m" + s + ESC + "[23m"

def blinking(s: str):
    return ESC + "[5m" + s + ESC + "[25m"

def warning(s: str):
    return yellow(s)

def ok(s: str):
    return green(s)


# Colors
colors = {
    "BLACK": ESC+"[90m",
    "RED": ESC+"[91m",
    "GREEN": ESC+"[92m",
    "YELLOW": ESC+"[93m",
    "BLUE": ESC+"[94m",
    "MAGENTA": ESC+"[95m",
    "CYAN": ESC+"[96m",
    "WHITE": ESC+"[97m",
    "DEFAULT": ESC+"[39m",
}
def blue(s: str):
    return colors['BLUE'] + s + colors['DEFAULT']

def yellow(s: str):
    return colors['YELLOW'] + s + colors['DEFAULT']

def red(s: str):
    return colors['RED'] + s + colors['DEFAULT']

def green(s: str):
    return colors['GREEN'] + s + colors['DEFAULT']

def magenta(s: str):
    return colors['MAGENTA'] + s + colors['DEFAULT']

def cyan(s: str):
    return colors['CYAN'] + s + colors['DEFAULT']

def white(s: str):  
    return colors['WHITE'] + s + colors['DEFAULT']

def black(s: str):
    return colors['BLACK'] + s + colors['DEFAULT']

def color_256(s: str, color: int):
    assert 0 <= color < 255
    return ESC + f"[38;5;{color}m" + s + colors['DEFAULT']

def color_rgb(s: str, rgb: Tuple[int, int, int]):
    r, g, b = rgb
    return ESC + f"[38;2;{r};{g};{b}m" + s + colors['DEFAULT']



if __name__ == "__main__":
    s = blinking(bold(italic(blue("Hello ") + color_rgb("world, ", (255, 150, 10)) + "nice to meet " + color_256("you", 150))))
    print(s)