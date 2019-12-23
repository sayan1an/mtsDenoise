import numpy as np
import matplotlib.pyplot as plt

fileName = "right1.txt"

f = open(fileName,"r")

lines = f.readlines()

nPoints = int(lines[0])
points = []
for i in range(0, nPoints):
    points.append(np.array(lines[i + 1].split(" ")).astype(np.float))
points = np.array(points)
offset = nPoints + 1

nIndices = int(lines[offset])
indices = []
for i in range(0, nIndices):
    indices.append(np.array(lines[i + offset + 1].split(" ")).astype(np.int))
indices = np.array(indices)
offset = offset + nIndices + 1

nSamples = int(lines[offset])
samples = []
for i in range(0, nSamples):
    samples.append(np.array(lines[i + offset + 1].split(" ")).astype(np.float))
samples = np.array(samples)

def drawLine(points, index0, index1):
    p0 = points[index0]
    p1 = points[index1]
    x1, y1 = [p0[0], p1[0]], [p0[1], p1[1]]
    plt.plot(x1, y1, marker = 'o', color="black")

for i in range(nIndices):
    tri = indices[i]
    drawLine(points, tri[0], tri[1])
    drawLine(points, tri[1], tri[2])
    drawLine(points, tri[2], tri[0])

for i in range(nSamples):
    sample = samples[i]
    if sample[2] > 0.5:
        plt.scatter(sample[0], sample[1], color="red")
    else:
        plt.scatter(sample[0], sample[1], color="blue")

plt.legend()
plt.savefig(fileName[0:-3] + "png")
plt.show()