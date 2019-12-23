import model
from scipy.ndimage.measurements import label
import cv2
import imageProcessing as ip
import numpy as np
import helper as aux
from moviepy.editor import ImageSequenceClip, VideoFileClip
import os.path
from tqdm import tqdm

class VehicleScanner:
    def __init__(self, imgInputShape=(720, 1280, 3), crop=(400, 660), pointSize=64,
                 confidenceThrd=.7, veHiDepth=30,
                 groupThrd=10, groupDiff=.1):

        self.crop = crop
        self.detectionPointSize = pointSize
        self.confidenceThrd = confidenceThrd

        bottomClip = imgInputShape[0] - crop[1]
        inH = imgInputShape[0] - crop[0] - bottomClip
        inW = imgInputShape[1]
        inCh = imgInputShape[2]

        self.cnnModel, cnnModelName = model.poolerPico(inputShape=(inH, inW, inCh))

        self.cnnModel.load_weights('{}.h5'.format(cnnModelName))

        self.veHiDepth = veHiDepth
        self.vehicleBoxesHistory = []
        self.groupThrd = groupThrd
        self.groupDiff = groupDiff

        self.diagKernel = [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]

    def vehicleScan(self, img):
        roi = img[self.crop[0]:self.crop[1], :]
        roiW, roiH = roi.shape[1], roi.shape[0]
        roi = np.expand_dims(roi, axis=0)
        detectionMap = self.cnnModel.predict(roi)
        predictionMapH, predictionMapW = detectionMap.shape[1], detectionMap.shape[2]
        ratioH, ratioW = roiH / predictionMapH, roiW / predictionMapW
        detectionMap = detectionMap.reshape(detectionMap.shape[1], detectionMap.shape[2])
        detectionMap = detectionMap > self.confidenceThrd
        labels = label(detectionMap, structure=self.diagKernel)
        hotPoints = []

        for vehicleID in range(labels[1]):
            nz = (labels[0] == vehicleID + 1).nonzero()
            nzY = np.array(nz[0])
            nzX = np.array(nz[1])
            xMin = np.min(nzX) - 32
            xMax = np.max(nzX) + 32
            yMin = np.min(nzY)
            yMax = np.max(nzY) + 64
            spanX = xMax - xMin
            spanY = yMax - yMin

            for x, y in zip(nzX, nzY):
                offsetX = (x - xMin) / spanX * self.detectionPointSize
                offsetY = (y - yMin) / spanY * self.detectionPointSize

                topLeftX = int(round(x * ratioW - offsetX, 0))
                topLeftY = int(round(y * ratioH - offsetY, 0))
                bottomLeftX = topLeftX + self.detectionPointSize
                bottomLeftY = topLeftY + self.detectionPointSize

                topLeft = (topLeftX, self.crop[0] + topLeftY)
                bottomRight = (bottomLeftX, self.crop[0] + bottomLeftY)

                hotPoints.append((topLeft, bottomRight))

        return hotPoints

    @staticmethod
    def addHeat(mask, bBoxes):
        for box in bBoxes:
            topY = box[0][1]
            bottomY = box[1][1]
            leftX = box[0][0]
            rightX = box[1][0]
            mask[topY:bottomY, leftX:rightX] += 1
            mask = np.clip(mask, 0, 255)
        return mask

    def getHotRegions(self, src):
        hotPoints = self.vehicleScan(img=src)
        sampleMask = np.zeros_like(src[:, :, 0]).astype(np.float)
        heatMap = self.addHeat(mask=sampleMask, bBoxes=hotPoints)
        currentFrameBoxes = label(heatMap, structure=self.diagKernel)

        return currentFrameBoxes, heatMap

    def updateHistory(self, currentLabels):
        for i in range(currentLabels[1]):
            nz = (currentLabels[0] == i + 1).nonzero()
            nzY = np.array(nz[0])
            nzX = np.array(nz[1])

            tlX = np.min(nzX)
            tlY = np.min(nzY)
            brX = np.max(nzX)
            brY = np.max(nzY)

            self.vehicleBoxesHistory.append([tlX, tlY, brX, brY])
            self.vehicleBoxesHistory = self.vehicleBoxesHistory[-self.veHiDepth:]

    def getRelevantBoxes(self, src):
        currentLabels, heatMapGray = self.getHotRegions(src=src)
        heatColor = aux.colorHeatMap(heatMapMono=heatMapGray, cmap=cv2.COLORMAP_JET)
        self.updateHistory(currentLabels=currentLabels)
        boxes, _ = cv2.groupRectangles(rectList=np.array(self.vehicleBoxesHistory).tolist(),
                                       groupThreshold=self.groupThrd, eps=self.groupDiff)
        return boxes, heatColor

class Detector:
    def __init__(self, imgMarginWidth=320, historyDepth=5, margin=100, windowSplit=2, winCount=9,
                 searchPortion=1., veHiDepth=30, pointSize=64,
                 groupThrd=10, groupDiff=.1, confidenceThrd=.7):
        self.imgProcessor = ip.Processing()
        self.warper = ip.Warping()
        self.imgMarginWidth = imgMarginWidth
        self.fitHistory = []
        self.scanner = VehicleScanner(pointSize=pointSize,
                                      veHiDepth=veHiDepth, groupThrd=groupThrd, groupDiff=groupDiff,
                                      confidenceThrd=confidenceThrd)

    def preProcess(self, src):
        imgHeight = src.shape[0]
        binary = ip.Thresholding.combiThreshold(src=src)
        filler = np.zeros((imgHeight, self.imgMarginWidth), dtype=np.uint8)
        binary = np.hstack((filler, binary, filler))
        binaryWarp = self.warper.birdEye(img=binary, leftShift=self.imgMarginWidth)
        return binaryWarp

    def addPip(self, pipImage, dstImage, pipAlpha=0.5, pipResizeRatio=0.3, origin=(20, 20)):
        smallPip = self.imgProcessor.resize(src=pipImage, ratio=pipResizeRatio)
        pipHeight = smallPip.shape[0]
        pipWidth = smallPip.shape[1]
        backGround = dstImage[origin[1]:origin[1] + pipHeight, origin[0]:origin[0] + pipWidth]
        blend = np.round(backGround * (1 - pipAlpha), 0) + np.round(smallPip * pipAlpha, 0)
        blend = np.minimum(blend, 255)
        dstImage[origin[1]:origin[1] + pipHeight, origin[0]:origin[0] + pipWidth] = blend

    def addOffsetStamp(self, leftFit, rightFit, image, origin, color=(255, 255, 255), fontScale=1.0, thickness=1):
        imgW = image.shape[1]
        imgH = image.shape[0]
        yBottom = imgH - 1
        cameraCenter = imgW / 2
        lBottomX = aux.funcSpace(argSpace=yBottom, fitParams=leftFit) - self.imgMarginWidth
        rBottomX = aux.funcSpace(argSpace=yBottom, fitParams=rightFit) - self.imgMarginWidth
        laneWidth = rBottomX - lBottomX
        scaleX = 3.7 / laneWidth
        laneCenter = (lBottomX + rBottomX) / 2
        offSet = (cameraCenter - laneCenter) * scaleX
        aux.putText(img=image, text='Смещение объекта автомобиля: {:.2f} m'.format(offSet),
                    origin=origin, color=color, scale=fontScale, thickness=thickness)

    def getEmbedDetections(self, src, pipParams=None):
        img = self.imgProcessor.undistort(src=src)
        vBoxes, heatMap = self.scanner.getRelevantBoxes(src=img)
        binary = self.preProcess(src=img)
        currFitLeft, leftFitType, leftBin = self.lineLeft.getFit(src=binary)
        currFitRight, rightFitType, rightBin = self.lineRight.getFit(src=binary)
        sanityPass = self.sanityCheckPass(currFitLeft, currFitRight)

        if sanityPass:
            img = self.addLanePoly(srcShape=binary.shape, dst=img, fitLeft=currFitLeft, fitRight=currFitRight)
        else:
            if self.lineLeft.reScanJustified():
                currFitLeft, leftFitType, leftBin = self.lineLeft.reScanWithPrimary(src=binary)
            else:
                self.lineLeft.resetFits()
                currFitLeft, leftFitType, leftBin = self.lineLeft.getFit(src=binary)
            if self.lineRight.reScanJustified():
                currFitRight, rightFitType, rightBin = self.lineRight.reScanWithPrimary(src=binary)
            else:
                self.lineRight.resetFits()
                currFitRight, rightFitType, rightBin = self.lineRight.getFit(src=binary)
            sanityPass = self.sanityCheckPass(currFitLeft, currFitRight)
            if sanityPass:
                img = self.addLanePoly(srcShape=binary.shape, dst=img, fitLeft=currFitLeft, fitRight=currFitRight)

        aux.drawBoxes(img=img, bBoxes=vBoxes)
        origin = (20, 20)

        if pipParams is not None:
            alpha = pipParams['alpha']
            ratio = pipParams['scaleRatio']
            commonBin = cv2.addWeighted(src1=leftBin, alpha=0.5, src2=rightBin, beta=0.5, gamma=1.0)
            pipHeight = int(commonBin.shape[0] * ratio)
            heatWidth = int(heatMap.shape[1] * ratio)
            self.addPip(pipImage=commonBin, dstImage=img,
                        pipAlpha=alpha, pipResizeRatio=ratio, origin=origin)

            self.addPip(pipImage=heatMap, dstImage=img,
                        pipAlpha=alpha, pipResizeRatio=ratio,
                        origin=(img.shape[1] - heatWidth - 20, 20))

            if currFitLeft is not None and currFitRight is not None:

                self.addOffsetStamp(leftFit=currFitLeft, rightFit=currFitRight,
                                    image=img, origin=(20, pipHeight + 70), fontScale=0.66, thickness=2,
                                    color=(0, 255, 0))
        return img


def main():
    resultFrames = []
    clipFileName = input('Введите имя видеофайла для обработки: ')

    if not os.path.isfile(clipFileName):
        print('Указанный файл не найден')
        return

    clip = VideoFileClip(clipFileName)

    depth = 5
    margin = 100
    fillerWidth = 320
    windowSplit = 2
    winCount = 18
    searchPortion = 1.

    pipAlpha = .7
    pipScaleRatio = .35

    pipParams = {'alpha': pipAlpha, 'scaleRatio': pipScaleRatio}

    ld = Detector(imgMarginWidth=fillerWidth, historyDepth=depth,
                  margin=margin, windowSplit=windowSplit, winCount=winCount,
                  searchPortion=searchPortion, veHiDepth=45,
                  pointSize=64, groupThrd=10, groupDiff=.1, confidenceThrd=.5)

    for frame in tqdm(clip.iter_frames()):
        dst = ld.getEmbedDetections(src=frame, pipParams=pipParams)
        resultFrames.append(dst)

    resultClip = ImageSequenceClip(resultFrames, fps=25, with_mask=False)
    resultFileName = clipFileName.split('.')[0]
    resultFileName = '{}_out_{}.mp4'.format(resultFileName, aux.timeStamp())
    resultClip.write_videofile(resultFileName, progress_bar=True)


if __name__ == '__main__':
    main()
