class LoadingBar:
    def __init__(self, total, bar_length=20):
        self.total = total
        self.bar_length = bar_length

    def update_progress(self, current):
        fraction = current / self.total
        arrow = int(fraction * self.bar_length - 1) * "-" + ">"
        padding = int(self.bar_length - len(arrow)) * " "
        ending = "\n" if current == self.total else "\r"
        print(f"Progress: [{arrow}{padding}] {int(fraction * 100)}%", end=ending)
