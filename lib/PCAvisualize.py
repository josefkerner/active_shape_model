import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import numpy as np
from matplotlib import pyplot as plt

class PCAvisualize:
    def __init__(self, pca):
        self.eigenvals = pca[0]
        self.eigenvecs = pca[1]
        self.mean = pca[2]

        #self.visualizeEigenVals()

        self.visualizeEigenVec()

        exit(1)

    def visualizeEigenVals(self):

        self.eigenvals = self.eigenvals.astype(int)

        print(self.eigenvals)
        tot = sum(self.eigenvals)
        var_exp = []

        for eigVal in sorted(self.eigenvals, reverse=True):
            exp = float(eigVal) / float(tot)
            var_exp.append(exp)

        cum_var_exp = np.cumsum(var_exp)

        trace1 = Bar(
                x=['PC %s' %i for i in range(1,11)],
                y=var_exp,
                showlegend=False)

        trace2 = Scatter(
                x=['PC %s' %i for i in range(1,11)],
                y=cum_var_exp,
                name='cumulative explained variance')

        data = Data([trace1, trace2])

        layout=Layout(
                yaxis=YAxis(title='Explained variance in percent'),
                title='Explained variance by different principal components')

        fig = Figure(data=data, layout=layout)
        py.iplot(fig)

    def visualizeEigenVec(self):

        from sklearn.manifold import TSNE

        n_sne = 7000

        eigenVectors = self.eigenvecs.astype(float)

        eigenVectors = eigenVectors.T
        eigenVectorsMin = min(eigenVectors[0])

        print('min',eigenVectorsMin)
        eigenVectorsMax = max(eigenVectors[0])

        plt.figure(1)

        plt.axis([eigenVectorsMin,eigenVectorsMax,eigenVectorsMin,eigenVectorsMax])


        for i in range(1,len(eigenVectors)):
            print(i)
            num = 200 + i
            print(num)
            plt.subplot(num)
            plt.plot(eigenVectors[i][0:319], eigenVectors[i][320:639])

        plt.show()

