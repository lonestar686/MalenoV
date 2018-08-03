import segyio
import numpy as np
import matplotlib.pyplot as plt

### ---- Functions for Input data(SEG-Y) formatting and reading ----
# Make a function that decompresses a segy-cube and creates a numpy array, and
# a dictionary with the specifications, like in-line range and time step length, etc.
def segy_decomp(segy_file, plot_data = False, read_direc='xline', inp_res = np.float64):
    # segy_file: filename of the segy-cube to be imported
    # plot_data: boolean that determines if a random xline should be plotted to test the reading
    # read_direc: which way the SEGY-cube should be read; 'xline', or 'inline'
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)

    # Make an empty object to hold the output data
    print('Starting SEG-Y decompressor')
    output = segyio.spec()

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        # Store some initial object attributes
        output.inl_start = segyfile.ilines[0]
        output.inl_end = segyfile.ilines[-1]
        output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]

        output.xl_start = segyfile.xlines[0]
        output.xl_end = segyfile.xlines[-1]
        output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]

        output.t_start = int(segyfile.samples[0])
        output.t_end = int(segyfile.samples[-1])
        output.t_step = int(segyfile.samples[1] - segyfile.samples[0])

        # for qc
        print(" SEGY inline/xline/t geometry:")
        print(" ils={}, ile={}, ili={}, ilen={}\n".format(output.inl_start, output.inl_end, 
                                                          output.inl_step,
                                                          segyfile.ilines.size))
                                                          
        print(" xls={}, xle={}, xli={}, xlen={}\n".format(output.xl_start, output.xl_end, 
                                                          output.xl_step,
                                                          segyfile.xlines.size))
        print(" ts={}, te={}, ti={}\n".format(output.t_start, output.t_end, output.t_step))

        # Pre-allocate a numpy array that holds the SEGY-cube
        output.data = np.empty((segyfile.ilines.size, segyfile.xlines.size, \
                               (output.t_end - output.t_start)//output.t_step+1), 
                               dtype = np.float32)

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for il_index in range(segyfile.ilines.size):     # WARNING: WEIRD!!!
                output.data[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'xline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for xl_index in range(segyfile.xlines.size):    # WARNING: WEIRD!!!
                output.data[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'full':
            ## NOTE: 'full' for some reason invokes float32 data
            # Potentially time this to find the "fast" direction
            #start = time.time()
            output.data = segyio.tools.cube(segy_file)
            #end = time.time()
            #print(end - start)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')


        # Convert the numpy array to span between -127 and 127 and convert to the desired format
        factor = 127/np.amax(np.absolute(output.data))
        if inp_res == np.float32:
            output.data = (output.data*factor)
        else:
            output.data = (output.data*factor).astype(dtype = inp_res)

        # If sepcified, plot a given x-line to test the read data
        if plot_data:
            # xline = 100
            xline = np.random.randint(output.data.shape[1])

            # Take a given xline
            data = output.data[:,xline,:]
            
            # Plot the read x-line
            plt.imshow(data.T,interpolation="nearest", cmap="gray")
            plt.title(' xline={}'.format(xline))
            plt.colorbar()
            plt.show()


    # Return the output object
    print('Finished using the SEG-Y decompressor')
    return output

# Make a function that adds another layer to a segy-cube
def segy_adder(segy_file, inp_cube, read_direc='xline', inp_res = np.float64):
    # segy_file: filename of the segy-cube to be imported
    # inp_cube: the existing cube that we should add a layer to
    # cube_num: which chronological number of cube is this
    # read_direc: which way the SEGY-cube should be read; 'xline', or 'inline'
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)

    # Make a variable to hold the shape of the input cube and preallocate a data holder
    print('Starting SEG-Y adder')
    cube_shape = inp_cube.shape
    dataholder = np.empty(cube_shape[0:-1])

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for il_index in range(segyfile.xline.len):
                dataholder[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'xline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for xl_index in range(segyfile.iline.len):
                dataholder[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'full':
            ## NOTE: 'full' for some reason invokes float32 data
            # Potentially time this to find the "fast" direction
            #start = time.time()
            dataholder[:,:,:] = segyio.tools.cube(segy_file)
            #end = time.time()
            #print(end - start)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')


        # Convert the numpy array to span between -127 and 127 and convert to the desired format
        factor = 127/np.amax(np.absolute(dataholder))
        if inp_res == np.float32:
            dataholder = (dataholder*factor)
        else:
            dataholder = (dataholder*factor).astype(dtype = inp_res)


    # Return the output object
    print('Finished adding a SEG-Y layer')
    return dataholder

# Convert a numpy-cube and seismic specs into a csv file/numpy-csv-format,
def csv_struct(inp_numpy,spec_obj,section,inp_res=np.float64,save=False,savename='default_write.ixz'):
    # inp_numpy: array that should be converted to csv
    # spec_obj: object containing the seismic specifications, like starting depth, inlines, etc.
    # inp_res: input resolution, the formatting of the seismic cube (could be changed to 8-bit data)
    # save: whether or not to save the output of the function
    # savename: what to name the newly saved csv-file

    # Get some initial parameters of the data
    (ilen,xlen,zlen) = inp_numpy.shape
    i = 0

    # Preallocate the array that we want to make
    full_np = np.empty((ilen*xlen*zlen,4),dtype = inp_res)

    # Itterate through the numpy-cube and convert each trace individually to a section of csv
    for il in range(section[0]*spec_obj.inl_step,(section[1]+1)*spec_obj.inl_step,spec_obj.inl_step):
        j = 0
        for xl in range(section[2]*spec_obj.xl_step,(section[3]+1)*spec_obj.xl_step,spec_obj.xl_step):
            # Make a list of the inline number, xline number, and depth for the given trace
            I = (il+spec_obj.inl_start)*(np.ones((zlen,1)))
            X = (xl+spec_obj.xl_start)*(np.ones((zlen,1)))
            Z = np.expand_dims(np.arange(section[4]*spec_obj.t_step+spec_obj.t_start,\
                                         (section[5]+1)*spec_obj.t_step+spec_obj.t_start,spec_obj.t_step),\
                               axis=1)

            # Store the predicted class/probability at each og the given depths of the trace
            D = np.expand_dims(inp_numpy[i,j,:],axis = 1)

            # Concatenate these lists together and insert them into the full array
            inp_li = np.concatenate((I,X,Z,D),axis=1)
            full_np[i*xlen*zlen+j*zlen:i*xlen*zlen+(j+1)*zlen,:] = inp_li
            j+=1
        i+=1

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename, full_np, fmt = '%f')

    # Return the list of adresses and classes as a numpy array
    return full_np

### ---- Functions for the training part of the program ----
# Make a function that combines the adress cubes and makes a list of class adresses
def convert(file_list, save = False, savename = 'adress_list', ex_adjust = False):
    """
    Arguments:
        file_list: list of file names(strings) of adresses for the different classes
        save: boolean that determines if a new ixz file should be saved with adresses and class numbers
        savename: desired name of new .ixz-file
        ex_adjust: boolean that determines if the amount of each class should be approximately equalized
    Return:
        list of (y, x, t, label)
    """


    # Make an array of that holds the number of each example provided, if equalization is needed
    if ex_adjust:
        len_array = np.zeros(len(file_list),dtype = np.float32)
        for i in range(len(file_list)):
            len_array[i] = len(np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.float32))

        # Cnvert this array to a multiplier that determines how many times a given class set needs to be
        len_array /= max(len_array)
        multiplier = 1//len_array


    # preallocate space for the adr_list, the output containing all the adresses and classes
    adr_list = np.empty([0,4], dtype = np.int32)

    # Itterate through the list of example adresses and store the class as an integer
    for i in range(len(file_list)):
        a = np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.int32)
        adr_list = np.append(adr_list,np.append(a,i*np.ones((len(a),1),dtype = np.int32),axis=1),axis=0)

        # If desired copy the entire list by the multiplier calculated
        if ex_adjust:
            for k in range(int(multiplier[i])-1):
                adr_list = np.append(adr_list,np.append(a,i*np.ones((len(a),1),dtype = np.int32),axis=1),axis=0)

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename + '.ixz', adr_list, fmt = '%i')

    # Return the list of adresses and classes as a numpy array
    return adr_list