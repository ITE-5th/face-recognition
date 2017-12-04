import cv2
import numpy as np

from CMS.minibatch_cms import CMSRCNN


def test():
    import os
    im_file = './123.jpg'
    image = cv2.imread(im_file)

    detector = CMSRCNN()
    detector.cuda()
    detector.eval()

    dets, scores, classes = detector.detect(image, 0.7)

    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)
    cv2.imshow('demo', im2show)
    cv2.waitKey(0)


if __name__ == '__main__':
    test()
