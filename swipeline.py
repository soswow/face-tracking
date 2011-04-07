import cv

from cvutils import *

V = "v"
H = "h"

class Event(object):
    def __init__(self, rect, open=False, close=False):
        self.rect = rect
        self.open = open
        self.close = close
        
        if open:
            self._repr = "open"
        elif close:
            self._repr = "close"
        else:
            self._repr = "None"

    def get_value(self, direction=H):
        if self.open:
            return self.rect.get_open(direction)
        else:
            return self.rect.get_close(direction)

    def __repr__(self):
        return "%s - %s" % (self.rect, self._repr)

class RectangleBundle(object):
    def __init__(self, rects):
        self.rects = rects
        [rect.deactivate() for rect in self.rects]

    def get_events(self, direction=H):
        queue = []
        rects = self.rects
        if direction == V:
            rects = self.active_rects(H)
        closes = sorted(rects, key=lambda k:k.get_close(direction))
        opens = sorted(rects, key=lambda k:k.get_open(direction))
        k=0
        while k != len(rects)*2:
            if opens and closes:
                open = opens[0].get_open(direction)
                close = closes[0].get_close(direction)
                if open < close:
                    queue.append(Event(opens.pop(0), open=True))
                elif open > close:
                    queue.append(Event(closes.pop(0), close=True))
            elif opens:
                queue += [Event(rect, open=True) for rect in opens]
                break
            else:
                queue += [Event(rect, close=True) for rect in closes]
                break
            k+=1
        return queue

    def active_rects(self, direction=H):
        return filter(lambda rect: rect.is_active(direction), self.rects)

    def queue_iterator(self, direction, func):
        queue = self.get_events(direction)

        prev, diff, total = 0, 0, 0
        for i, event in enumerate(queue):
            value = event.get_value(direction)
            if i > 0:
                diff = value - prev

            total += func(diff) if diff else 0

            if event.open:
                event.rect.activate(direction)
            else: #right
                event.rect.deactivate(direction)

            prev = value
        return total

    def vertical_event_handler(self, diff):
        return diff if self.active_rects(V) else 0

    def horizontal_event_handler(self, diff):
        if not self.active_rects(H):
            return 0
        v_sum = self.queue_iterator(V, self.vertical_event_handler)
        return v_sum * diff

    def rect_area(self):
        return self.queue_iterator(H, self.horizontal_event_handler)

    def draw(self, img):
        for r in self.rects:
            cv.Rectangle(img,(r.left,r.top), (r.right, r.bottom), 255, thickness=1)

class Rectangle(object):
    def __init__(self,x,y,w,h):
        self.w = w
        self.h = h
        self.top = y
        self.bottom = y+h
        self.left = x
        self.right = x+w
        self.active = {H:False, V:False}

    def get_open(self, direction=H):
        return self.left if direction == H else self.top

    def get_close(self, direction=H):
        return self.right if direction == H else self.bottom

    def is_active(self, direction=H):
        return self.active[direction]

    def activate(self, direction=H):
        self.active[direction] = True

    def deactivate(self, direction=H):
        self.active[direction] = False

    def __repr__(self):
        return "Rect(%d,%d,%d,%d)" % (self.left, self.top, self.w, self.h)

def calculate_area(tuple_list):
    return build_rectangle_bundle(tuple_list).rect_area()

def build_rectangle_bundle(tuple_list):
    return RectangleBundle([Rectangle(x,y,w,h) for x,y,w,h in tuple_list])

def main():
    img = cv.CreateImage((500,500),8,1)
    cv.Zero(img)
    area = calculate_area([(20, 50, 20, 100),
                         (50, 50, 100, 100),
                         (70, 70, 100, 100),
                         (130, 70, 100, 100),
                         (130, 90, 60, 200),
                         (90, 190, 100, 40),
                         (200, 30, 20, 300),
                         (260, 80, 40, 40)])
#    area = bundle.rect_area()
    print "Area is %d" % area
    assert area == 36000

#    bundle.draw(img)
#    show_image(img)



if __name__ == "__main__":
    main()