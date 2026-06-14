import curses

TUI_BACKGROUND_PAIR_ID = 1
TUI_HEADING_PAIR_ID = 2

TUI_TEXT = 1
TUI_HEADING = 2

class TUI:
    def __init__(self, screen: curses.window):
        self.screen: curses.window = screen
        self.screen.clear()
        
        # Configure color combinations
        curses.init_pair(TUI_BACKGROUND_PAIR_ID, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(TUI_HEADING_PAIR_ID, curses.COLOR_YELLOW, curses.COLOR_BLACK)

        # Set the background color
        self.screen.bkgdset(' ', curses.color_pair(TUI_BACKGROUND_PAIR_ID))

        self.row: int = 0
        self.col: int = 0
    
    def print(self, string, style: int=TUI_TEXT):
        # Load attributes from text style
        attr = 0
        if style == TUI_TEXT:
            attr = 0
        elif style == TUI_HEADING:
            attr = curses.A_BOLD | curses.color_pair(TUI_HEADING_PAIR_ID)
        
        # Print the string to the screen
        self.screen.addstr(self.row, self.col, string, attr)
        self.row += 1
    
    def set_col(self, col):
        self.col = col
    
    def set_row(self, row):
        self.row = row
    
    def skip_rows(self, n):
        self.row += n
    
    def update(self):
        self.screen.refresh()
    
    # Wait until any key is pressed, then continue.
    def wait(self, msg: str | None=None) -> int:
        if msg is None:
            msg = "Waiting for confirmation to continue. Press any key to continue:"

        # Print confirmation message and get the character typed
        self.row += 1
        self.print(msg + ' ')
        ch = self.screen.getch() # Main wait

        # Clear the confirmation message
        self.row -= 1
        self.screen.move(self.row, 0)
        self.screen.clrtoeol()
        self.row -= 1
        self.screen.move(self.row, 0)
        self.screen.refresh()

        return ch