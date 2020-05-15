import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
def sse(centroids, pts):
    sum = 0
    for i, centroid in enumerate(centroids):
        for pt in pts[i]:
            sum += math.sqrt((centroid[0] - pt[0])**2 + (centroid[1] - pt[1])**2)
    return sum

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

points = []
with open('clu_data.txt') as file:
  for line in file:
    xtemp, ytemp = line.split(' ')
    points.append((float(xtemp), float(ytemp)))


minX, maxX = min(points)[0], max(points)[0]
minY, maxY = min(points)[1], max(points)[1]
print(f"minX = {minX}, MaxX = {maxX}, minY = {minY}, maxY = {maxY} ")

epochs = 40
colors = ['maroon', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'gold', 'teal', 'brown', 'coral', 'khaki', 'lime', 'magenta', 'olive', 'navy', 'lavender', 'grey', 'gold', 'darkgreen', 'silver', 'wheat', 'orchid', 'pink', 'purple', 'tan', 'sienna']
ax = plt.gca()
threshold = 0.02
sse_data = {}
KmeansRange = range(2, 22)
for K in KmeansRange:
    sse_data[K] = []
    for trial in range(10):
        converged = False
        Ks = []
        clusters = {}
        for k in range(K):
            x = random.uniform(minX, maxX)
            y = random.uniform(minY, maxY)
            Ks.append((x, y))

        for epoch in range(epochs):
            if converged:
                s = f"output/K-{K}-trial-{trial+1}-converged-{epoch+1}.png"
                plt.savefig(s)
                print(f"K = {K}, Trial {trial}, Converged by epoch {epoch}")
                print(f"K={K}, Trial #{trial+1}, Epoch #{epoch+1}, SSE={sse(Ks, clusters)}")
                # plt.pause(1)
                plt.close()
                break
            else:
                plt.close()
            for k in range(K):
                clusters[k] = []

            for i in range(len(points)):
                closestCluster = 0
                d = distance(Ks[0], points[i])
                for k in range(1, K):
                    dTemp = distance(Ks[k], points[i])
                    if dTemp < d:
                        d = dTemp
                        closestCluster = k
                clusters[closestCluster].append(points[i])
            for k in range(K):
                x_val = [x[0] for x in clusters[k]]
                y_val = [x[1] for x in clusters[k]]
                color = next(ax._get_lines.prop_cycler)['color']
                plt.xlabel("x")
                plt.ylabel("y")
                plt.scatter(x_val,y_val, color=colors[k])
                plt.scatter(Ks[k][0], Ks[k][1], marker='+', color=colors[k])
            sse_data[K].append(sse(Ks, clusters))
            plt.title(f"K={K}, Trial #{trial+1}, Epoch #{epoch+1}, SSE={sse_data[K][-1]:.2f}")
            plt.show(block=False)
            converged = True
            for k in range(K):
                if len(clusters[k])> 0:
                    mean = np.mean(clusters[k], axis=0)
                    mean = (mean[0], mean[1])
                    if distance(mean, Ks[k]) > threshold:
                        converged = False
                    Ks[k] = mean

ks = []
sses = []
for K in KmeansRange:
    print(f"{K} -- {min(sse_data[K])}")
    ks.append(K)
    sses.append(min(sse_data[K]))
plt.clf()
plt.title("Sum Square Errors vs K")
plt.xlabel("K")
plt.ylabel("SSE")
plt.scatter(ks, sses , color=colors[0])
print(ks)
print(sses)
s = f"output/ScatterSSEoverBestKs-{KmeansRange[0]}-{KmeansRange[-1]}.png"
plt.savefig(s)
plt.show()


plt.clf()
plt.title("Sum Square Errors vs K")
plt.xlabel("K")
plt.ylabel("SSE")
plt.plot(ks, sses , color=colors[0])
s = f"output/PlotSSEoverBestKs-{KmeansRange[0]}-{KmeansRange[-1]}.png"
plt.savefig(s)
plt.show()