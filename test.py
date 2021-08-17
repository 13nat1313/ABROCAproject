import numpy as np
import datautility as du


def check_args():
    normal_count = 0
    for val in cov_dist:
        if val == 0:
            normal_count += 1
    if len(means) != normal_count:
        quit('means and normal_count not aligned')
    if len(stdevs) != normal_count:
        quit('stdevs and normal_count not aligned')
    for key in means:
        if len(means[key]) != sum(num_entities):
            quit('length of means columns incorrect')
    for key in stdevs:
        if len(stdevs[key]) != sum(num_entities):
            quit('length of stdevs columns incorrect')
    if len(group_mean_adjustment) != num_groups:
        quit('mean adjustment and num_groups not aligned')
    if len(group_stdev_adjustment) != num_groups:
        quit('stdev adjustment and num_groups not aligned')


def get_normal_value(mean, stdev):
    return np.random.normal(loc=int(mean), scale=int(stdev))


def get_custom_value(filename, header):
    data, headers = du.read_csv(filename)
    column = headers.index(header)
    vals = [row[column] for row in data]
    return np.random.choice(vals)


def simulate_experiment():
    exp_vals = {}
    for i, e in enumerate(num_entities):
        exp_vals['group'+str(i+1)] = {}
        for j, mkey in enumerate(means):
            exp_vals['group' + str(i + 1)][mkey] = []
            for k in range(e):
                if cov_dist[j] == 0:
                    mean = int(means[mkey][k]) + group_mean_adjustment[i]
                    stdev = int(stdevs[mkey][k]) + group_stdev_adjustment[i]
                    exp_vals['group' + str(i + 1)][mkey].append(get_normal_value(mean, stdev))
                if cov_dist[j] == 1:
                    exp_vals['group' + str(i + 1)][mkey].append(get_custom_value('resources/CustomDistribution.csv', 'Values'))
    return exp_vals


if __name__ == '__main__':
    num_entities = [50, 50]
    num_groups = len(num_entities)

    cov_dist = [0, 0, 0, 0]
    num_covariates = len(cov_dist) - 1

    group_mean_adjustment = [-50, 150]
    group_stdev_adjustment = [0, 0]

    means = {}
    stdevs = {}

    mean_data, mh = du.read_csv('resources/means.csv')
    stdev_data, sh = du.read_csv('resources/stdevs.csv')

    for h, s in zip(mh, sh):
        if h != s:
            quit('headers dont match')

        means[h.strip().replace('\ufeff', '')] = []
        stdevs[s.strip().replace('\ufeff', '')] = []

    for mrow, srow in zip(mean_data, stdev_data):
        if len(mrow) != len(srow):
            quit('row lengths dont match')
        for col in range(len(mrow[:-1])):
            means['covariate'+str(col+1)].append(mrow[col])
            stdevs['covariate' + str(col+1)].append(srow[col])
        means['errors'].append(mrow[-1])
        stdevs['errors'].append(srow[-1])

    check_args()
    all_values = simulate_experiment()
    print(all_values)
    for group_key in all_values:
        print(group_key)
        for key in all_values[group_key]:
            all_values[group_key][key] = [('mean', np.mean(all_values[group_key][key])), ('std', np.std(all_values[group_key][key])), ('sum', np.sum(all_values[group_key][key]))]
            print("   "+str(key))
            print("   "+str(all_values[group_key][key]))