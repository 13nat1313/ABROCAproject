import numpy as np
import os
import sys
import csv
import warnings


def getfilenames(directory='./', extension=None, exclude_directory=False):
    names = []
    directory = str(directory).replace('\\','/')
    if extension is None:
        return os.listdir(directory)

    for file in os.listdir(directory):
        if file.endswith(extension):
            if exclude_directory:
                names.append(file)
            else:
                names.append(directory + ('/' if directory[-1] != '/' else '') + file)
    return names


def file_exists(filename, directory='./',extension=None):
    if extension is None:
        extension = ''
    if filename[len(filename)-len(extension):] != extension:
        filename = filename + extension
    return filename in getfilenames(directory,extension,exclude_directory=True)


def read_paired_data_file(filename, delim='=', ignore='--'):
    data = dict()

    with open(filename, 'r', errors='replace') as f:
        info = np.array(f.readlines())

        if ignore is not None:
            info = [i.strip() for i in
                    info[np.argwhere([j not in ['\n'] and not j.startswith(ignore) for j in info]).reshape((-1))]]

        for i in info:
            pair = [j.strip() for j in i.split(delim)]
            if len(pair) != 2:
                print('\n\033[91m', 'ERROR - Unable to parse the line: {}'.format(i), '\033[0m\n')
                sys.stdout.flush()
            else:
                data[pair[0].lower()] = pair[1]

    return data


def read_text_file(filename, sep=None, ignore='--', max_rows=None):
    text = '' if sep is not None else []

    n_lines = len(open(filename).readlines())
    with open(filename, 'r', errors='replace') as f:

        info = None
        if max_rows is not None:
            info = []
            for i in range(min(max_rows,n_lines)):
                info.append(f.readline())
            info = np.array(info)
        else:
            info = np.array(f.readlines())

        info[0] = str(info[0]).replace('ï»¿', '')

        if ignore is not None:
            info = [i.strip() for i in
                    info[np.argwhere([j not in ['\n'] and not j.startswith(ignore) for j in info]).reshape((-1))]]

        for i in info:
            if sep is not None:
                text += i + sep
            else:
                text.append(i)

    return text


def __load_csv__(filename, max_rows=None, columns=None, encoding=None, verbose=True):
    csvarr = []
    try:
        n_lines = len(open(filename, encoding=encoding).readlines())
    except (TypeError, UnicodeDecodeError):
        encoding = 'utf8'
        n_lines = len(open(filename, encoding=encoding).readlines())
    with open(filename, 'r', errors='replace', encoding=encoding) as f:
        f_lines = csv.reader(f)

        # n_lines = sum(1 for row in f_lines)
        # print(n_lines)
        # n_lines = sum(1 for row in f_lines)
        if max_rows is not None:
            n_lines = max_rows

        if verbose:
            output_str = '-- loading {}...({}%)'.format(filename, 0)
            sys.stdout.write(output_str)
            sys.stdout.flush()
            old_str = output_str
        i = 0
        filter = None
        for line in f_lines:
            if len(line) == 0:
                continue

            if filter is None:
                if columns is None:
                    filter = np.array(range(len(line)),dtype='int32')
                else:
                    if not isinstance(columns,list):
                        filter = np.array([columns])
                    if isinstance(columns[0],str):
                        filter = np.array([np.argwhere(np.array(line).ravel() == c).ravel()[0] for c in columns])
                    else:
                        filter = np.array(columns,dtype='int32')
                filter = filter.ravel()

            line = np.array(line)[filter].ravel()
            na = np.argwhere(np.array(line[:]) == '#N/A').ravel()
            if len(na) > 0:
                line[na] = ''

            na = np.argwhere(np.array(line[:]) == 'NA').ravel()

            if len(na) > 0:
                line[na] = ''

            csvarr.append(line)
            if max_rows is not None:
                if len(csvarr) >= max_rows:
                    break
            if not round((i / n_lines) * 100, 2) == round(((i - 1) / n_lines) * 100, 2):
                if verbose:
                    sys.stdout.write('\r' + (' ' * len(old_str)))
                    output_str = '\r-- loading {}...({}%)'.format(filename, round((i / n_lines) * 100, 2))
                    sys.stdout.write(output_str)
                    sys.stdout.flush()
                    old_str = output_str

            i += 1

        if verbose:
            sys.stdout.write('\r' + (' ' * len(old_str)))
            sys.stdout.write('\r-- loading {}...({}%)\n'.format(filename, 100))
            sys.stdout.flush()

    return csvarr


def write_csv(data, filename, headers=None, append=False, encoding=None):
    if append:
        headers = None
    if headers is None:
        headers = []

    if not filename.endswith('.csv'):
        filename += '.csv'

    try:
        _ = open(filename, 'w' if not append else 'a', errors='replace', encoding=encoding)
    except UnicodeEncodeError:
        encoding = 'UTF-8'

    with open(filename, 'w' if not append else 'a', errors='replace', encoding=encoding) as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')

        if len(headers)!=0:
            writer.writerow(np.array(headers, dtype=str))
            # for i in range(0,len(headers)-1):
            #     f.write(str(headers[i]) + ',')
            # f.write(str(headers[len(headers)-1])+'\n')
        # for i in range(0,len(data)):
        # ar = np.array(data, dtype=str)
        # ar = ar.reshape((ar.shape[0],-1))

        for j in data:
            row = np.array(j, dtype=str)
            row[np.argwhere([k == 'None' for k in row]).ravel()] = ''
            row[np.argwhere([k is None for k in row]).ravel()] = ''
            writer.writerow(row)

    f.close()


def read_csv(filename, max_rows=None, columns=None, headers=True, encoding=None, verbose=True):
    if max_rows is not None:
        max_rows += 1

    if not filename.endswith('.csv'):
        filename += '.csv'

    data = __load_csv__(filename,max_rows,columns,encoding,verbose)

    if headers:
        headers = np.array(data[0])
        data = np.delete(data, 0, 0)
        return data, headers
    else:
        return data


def read_csv_headers(filename):
    if not filename.endswith('.csv'):
        filename += '.csv'

    with open(filename, 'r') as f:
        for line in f.readlines():
            return line.strip().split(',')
    return []



def infer_if_string(ar, n=None):
    ar = np.array(ar)
    assert len(ar.shape) == 1

    if n is None:
        n = ar.shape[0]
    else:
        n = np.minimum(ar.shape[0],n)

    for i in range(n):
        try:
            float(ar[i])
        except ValueError:
            if ar[i] == '':
                continue
            else:
                return True
    return False


def infer_basic_type(ar, n=None):
    ar = np.array(ar)
    assert len(ar.shape) == 1

    if n is None:
        n = ar.shape[0]
    else:
        n = np.minimum(ar.shape[0],n)

    is_int = True

    for i in range(n):
        try:
            temp = float(ar[i])
            if not temp == int(temp):
                is_int = False
        except ValueError:
            if ar[i] == '':
                continue
            else:
                return 'text'
        except TypeError:
            return 'obj'
    return 'double precision' if not is_int else 'integer'


def as_factor(ar, return_labels=False):
    ar = np.array(ar).reshape((-1))
    label = np.unique(ar)
    for i in range(len(label)):
        ar[ar[:] == label[i]] = i
    if return_labels:
        lab = [label[int(i)] for i in ar]
        return ar, lab
    return ar


def as_float(ar):
    ar = np.array(ar, dtype=str)
    ar[ar == '.'] = float('nan')
    ar[ar == ' '] = float('nan')
    ar[ar == ''] = float('nan')
    return np.array(ar, dtype=np.float32).reshape((-1))


def print_descriptives(ar, headers=None, desc_level=1):
    ar = np.array(ar)
    ar = ar.reshape((-1,ar.shape[-1]))

    if headers is not None:
        # assert len(headers) == ar.shape[-1]
        headers = [str(i) + ' ' + headers[i] if i < len(headers) else 'Covariate ' + str(i) for i in
                   range(ar.shape[-1])]
    else:
        headers = ['Covariate ' + str(i) for i in range(ar.shape[-1])]

    print("{:=<{size}}".format('', size=50 + (30 * desc_level)))
    print("{:<15}{:^25}".format('DESCRIPTIVES', "{} Rows, {} Columns".format(ar.shape[0],ar.shape[1])))
    print("{:=<{size}}".format('', size=50 + (30 * desc_level)))
    for i in range(ar.shape[-1]):
        h = headers[i]
        if len(h) > 15:
            h = ''.join(list(h)[:15]) + '...'
        label = "Column {}".format(i) if headers is None else h
        # print(ar[0, i])
        dtype = ['int','float','string','obj'][np.array(np.where(
            np.array(['integer','double precision','text','obj'])[:] == infer_basic_type(ar[:, i], 1000))).reshape((-1))[0]]
        label = "{} ({}):".format(label,dtype)

        if dtype == 'string':
            m = np.array(np.array(ar[:, i]) == '').sum()
            desc1 = "{} unique values".format(len(np.unique(np.array(ar[:, i],dtype=str))))
            desc2 = ''
            desc3 = ''
        elif dtype == 'obj':
            desc1 = "{} data type".format(type(ar[0, i]))
            desc2 = ''
            desc3 = ''
        else:
            ar[:, i][ar[:, i] == ''] = float('nan')
            f_ar = np.array(ar[:, i], dtype=np.float32)
            m = np.isnan(f_ar).sum()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                desc1 = "mean={:<.2f} (SD={:<.2f})".format(np.nanmean(f_ar), np.nanstd(f_ar))
                desc2 = "median = {:<.2f}".format(np.nanmedian(f_ar))
                desc3 = "min={:<.2f}, max={:<.2f}".format(np.nanmin(f_ar), np.nanmax(f_ar))
        missing = "{} missing ({:<.1f}%)".format(m, m / float(ar.shape[0]))
        print("{:<30} {:<20} {:<35} {:<30} {:<30}".format(label, missing,
                                                          desc1 if desc_level > 0 else '',
                                                          desc2 if desc_level > 1 else '',
                                                          desc3 if desc_level > 2 else ''))
    print("{:=<{size}}\n".format('', size=50 + (30 * desc_level)))


def ndims(ar):
    d = 0
    a = ar

    while hasattr(a, '__iter__'):
        d += 1
        a = a[0]

        try:
            _ = '0' + a
            break
        except TypeError:
            pass

    return d

