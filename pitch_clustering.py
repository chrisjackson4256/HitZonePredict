import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def pitch_clustering(df):
    '''
    Function to perform K-means cluster analysis and return a "pitch vector" for
    each pitcher which contains the frequency that the pitcher throws each pitch type
    input: df - dataframe of pitch characteristics (speed, location, spin, etc.)
    output: dataframe with k columns (where k is determined through k-means clustering
            and the "Gap Statistic" to determine the optimal number of clusters)
    '''
    # rescale the training data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df))
    df_scaled.columns = df.columns

    # fitting the PCA algorithm with our Data
    pca = PCA().fit(df_scaled)

    # the cumulative explained variance
    exp_var_ratio = list(np.cumsum(pca.explained_variance_ratio_))

    # use the number of components that get us to 98% explained variance
    exp_var_threshold = 0.98
    for i, ev in enumerate(exp_var_ratio):
        if ev > exp_var_threshold:
            num_components = i
            break
        else:
            num_components = len(df_scaled.columns)

    # in case the explained variance never gets to threshold, let the user know
    if num_components == len(df_scaled.columns):
        print(f"Explained variance ratio never reaches threshold of {exp_var_threshold}")

    # let the user know how many components PCA selected
    print(f"Number of components from PCA: {num_components}")

    # transform the data to the selected number of dimensions
    pca = PCA(n_components=7)
    df_pca = pd.DataFrame(pca.fit_transform(df_scaled))

    # perform k-means clustering and use the Gap Statistic to find optimal k
    def optimalK(data, nrefs=3, maxClusters=15):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, maxClusters)):

            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)

            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):

                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)

                # Fit to it
                km = KMeans(k, n_jobs=-1)
                km.fit(randomReference)

                refDisp = km.inertia_
                refDisps[i] = refDisp

            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)

            origDisp = km.inertia_

            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)

            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap

            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

        return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

    k, _ = optimalK(df_pca)
    print(f"The optimal number of clusters is: {k}")
