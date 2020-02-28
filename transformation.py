'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import matplotlib.pyplot as plt
import analysis
import data
import palettable


class Transformation(analysis.Analysis):

    def __init__(self, data_orig, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        data_orig: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables
            — `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variables for `data_orig`.
        '''
        self.data_orig = data_orig
        # self.data = super().__init__(data)
        self.data = data
        analysis.Analysis.__init__(self,self.data)
        pass

    def project(self, headers):
        '''Project the data on the list of data variables specified by `headers` — i.e. select a
        subset of the variables from the original dataset. In other words, populate the instance
        variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list dictates the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables are optional.

        HINT: Update self.data with a new Data object and fill in appropriate optional parameters
        (except for `filepath`)

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables).
        - Make sure that you create 'valid' values for all the `Data` constructor optional parameters
        (except you dont need `filepath` because it is not relevant).
        '''
        header2col = {}
        for i in range(len(headers)):
            header2col[headers[i]] = i
        self.data = data.Data(headers = headers, data = self.data_orig.select_data(headers), header2col=header2col)
        pass

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        ones = np.ones([self.data.data.shape[0], 1])
        return np.hstack((self.data.data, ones))
        pass

    def translation_matrix(self, headers, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        identity = np.eye(len(headers)+1)
        magnitudes = np.hstack((magnitudes, [1])) # to match the dimension
        identity[:,-1] = magnitudes
        return identity
        pass

    def scale_matrix(self, headers, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        identity = np.eye(len(headers) + 1)
        magnitudes = np.hstack((magnitudes, [1])) # to match the dimension
        return identity * np.array([magnitudes]).T
        pass

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data about the ONE
        axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        identity = np.eye(4)
        angle = np.radians(degrees)
        if (self.data.headers.index(header) == 0):
            identity[1,1] = np.cos(angle)
            identity[2,2] = np.cos(angle)
            identity[1,2] = -np.sin(angle)
            identity[2,1] = np.sin(angle)
        elif (self.data.headers.index(header) == 1):
            identity[0,0] = np.cos(angle)
            identity[0,2] = np.sin(angle)
            identity[2,0] = -np.sin(angle)
            identity[2,2] = np.cos(angle)
        else:
            identity[0,0] = np.cos(angle)
            identity[0,1] = -np.sin(angle)
            identity[1,0] = np.sin(angle)
            identity[1,1] = np.cos(angle)
        return identity
        pass

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected dataset after it has been transformed by `C`
        '''
        return (C @ self.get_data_homogeneous().T).T
        pass

    def translate(self, headers, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be translated.
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        # take off one column
        ret = self.transform(self.translation_matrix(headers, magnitudes))[:,:-1]
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings)
        return ret
        pass

    def scale(self, headers, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        headers: Python list of str.
            Specifies the variables along which the projected dataset should be scaled.
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        # take off one column
        ret = self.transform(self.scale_matrix(headers, magnitudes))[:,:-1]
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings())
        return ret
        pass

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplcation to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        # take off one column
        ret = self.transform(self.rotation_matrix_3d(header,degrees))[:,:-1]
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings())
        return ret
        pass

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        globalMin = np.min(self.data.data)
        globalMax = np.max(self.data.data)
        globalRange = globalMax - globalMin
        ret = (self.data.data - globalMin) / globalRange
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings())
        return ret
        pass

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        localMin = np.min(self.data.data, axis = 0)
        localMax = np.max(self.data.data, axis = 0)
        localRange = localMax - localMin
        ret = (self.data.data - np.array([localMin,] * self.data.data.shape[0])) / np.array([localRange,] * self.data.data.shape[0])
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings())
        return ret
        pass

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Section B (Linear Algebra): Use a ColorBrewer color palette (e.g. from the
            `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        fig, ax = plt.subplots()

        # set the elements of plot
        if title != None:
            ax.set_title(title)
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)

        x = self.data.select_data(ind_var)
        y = self.data.select_data(dep_var)
        color = self.data.select_data(c_var)
        pos = ax.scatter(x, y, c=color, cmap=palettable.colorbrewer.sequential.Greys_5.mpl_colormap)
        cb = fig.colorbar(pos, ax=ax)
        cb.set_label(c_var)

        pass

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap)

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls )
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax


    def normalize_zscore(self):
        '''Normalize each variable subtracting the mean value and then divide by the standard deviation.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.
        '''
        mean = np.sum(self.data.data) / self.data.data.size
        # to calculate the std
        nominator = np.square(np.subtract(self.data.data, mean))
        std = np.sum(nominator) / nominator.size

        ret = np.subtract(self.data.data, mean) / std
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings())
        return ret
        pass


    def rotation_matrix_2d(self, degrees):
        '''Make an 2-D homogeneous rotation matrix for rotating the projected data x degrees.

        Parameters:
        -----------
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(2, 2). The 2D rotation matrix.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        identity = np.eye(2)
        angle = np.radians(degrees)
        identity[0,0] = np.cos(angle)
        identity[0,1] = -np.sin(angle)
        identity[1,0] = np.sin(angle)
        identity[1,1] = np.cos(angle)
        return identity
        pass


    def rotate_2d(self, degrees):
        '''Rotates the projected data by the angle (in degrees).

        Parameters:
        -----------
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset.
        '''
        # take off one column
        ret = (self.rotation_matrix_2d(degrees) @ self.data.data.T).T
        self.data = data.Data(headers=self.data.headers, data=ret, header2col=self.data.get_mappings())
        return ret
        pass

