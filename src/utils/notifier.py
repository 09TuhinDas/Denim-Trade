from pync import Notifier

def notify(title, message):
    Notifier.notify(message, title=title)
