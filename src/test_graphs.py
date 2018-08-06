#!/usr/bin/env python
import graph
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import distance
import numpy as np
import pickle
from time import time

def get_min_max_dist(point1, point2, points):
    min_1 = (-99,1000)
    min_2 = (-99,1000)
    for i in range(len(points)):
        if distance.euclidean(point1,points[i]) < min_1[1]:
            min_1 = (i,distance.euclidean(point1,points[i]))
        if distance.euclidean(point2,points[i]) < min_2[1]:
            min_2 = (i,distance.euclidean(point2,points[i]))
    return (min_1[0],min_2[0])
def run_main():
    graph.plt.show(block=False)
    # Start by testing the variations due to size of the graph
    cur_runs = [1,2,3,4,5]
    # For each size in runs, 100 graphs will be generated using that size as
    # the number of nodes in the graph. Each graph will also have a blowout of
    # 50%, the area is proportional to the number of nodes.
    #
    # For each graph, each type of policy will be generated and simulated 100
    # 100 times. The average score and variance will be compared against the
    # baseline (simple), and that performance will be logged for that graph.
    # Once all 100 graphs have been evaluated at that size, a data point will be
    # added to each of the lists below, each of the form (mean,std_dev), where
    # the mean is the mean performance against the baseline for the 100 graphs,
    # and std_dev is the standard deviation of performances for the 100 graphs.
    # The two performance metrics considered are the mean score of the policy on
    # the graph, and the standard deviation of the score of the policy on the graph.

    # Notes: Size does not seem to have a large impact on how the different
    # modes perform, so we will try changing connectivity on two runs:
    try:
        simple_performances = pickle.load(open('simple_p.p','rb'))
        simple_conservative_performances = pickle.load(open('simple_c_p.p','rb'))
        multimodal_performances = pickle.load(open('multimodal_p.p','rb'))
        multimodal_conservative_performances = pickle.load(open('multimodal_c_p.p','rb'))
        simple_variances = pickle.load(open('simple_v.p','rb'))
        simple_conservative_variances = pickle.load(open('simple_c_v.p','rb'))
        multimodal_variances = pickle.load(open('multimodal_v.p','rb'))
        multimodal_conservative_variances = pickle.load(open('multimodal_c_v.p','rb'))
        runs = pickle.load(open('runs.p','rb'))
    except IOError as e:
        simple_performances = []
        simple_conservative_performances = []
        multimodal_performances = []
        multimodal_conservative_performances = []
        simple_variances = []
        simple_conservative_variances = []
        multimodal_variances = []
        multimodal_conservative_variances = []
        runs = []
    for run in cur_runs:
        runs.append(run)
        simple_performance_data = []
        simple_conservative_performance_data = []
        multimodal_performance_data = []
        multimodal_conservative_performance_data = []
        simple_variance_data = []
        simple_conservative_variance_data = []
        multimodal_variance_data = []
        multimodal_conservative_variance_data = []
        for i in range(100):
            print("{}% of upper_bound {}".format(i,run))
            g = graph.multimodal_graph()
            g.generate(vertices = 100, multiplier = run, blowout = .25, num_modes = 2)
            start,end = get_min_max_dist((-100,-100),(100,100),g.points)
            g.draw_graph()
            pi1 = g.gen_policy('simple', end)
            pi2 = g.gen_policy('simple_conservative', end)
            pi3 = g.gen_policy('multimodal', end)
            pi4 = g.gen_policy('multimodal_conservative', end)
            g.draw_policy(pi1,start,end,lnspec='r-')
            g.draw_policy(pi2,start,end,lnspec='b-')
            g.draw_policy(pi3,start,end,lnspec='g-')
            g.draw_policy(pi4,start,end,lnspec='c-')
            graph.plt.pause(1)
            graph.plt.clf()
            pi1_scores = np.array(g.simulate_policy(pi1,start, 100))
            pi2_scores = np.array(g.simulate_policy(pi2,start, 100))
            pi3_scores = np.array(g.simulate_policy(pi3,start, 100))
            pi4_scores = np.array(g.simulate_policy(pi4,start, 100))
            pi1_mean = np.mean(pi1_scores)
            pi2_mean = np.mean(pi2_scores)
            pi3_mean = np.mean(pi3_scores)
            pi4_mean = np.mean(pi4_scores)
            pi1_std = np.std(pi1_scores)
            pi2_std = np.std(pi2_scores)
            pi3_std = np.std(pi3_scores)
            pi4_std = np.std(pi4_scores)
            simple_performance_data.append(pi1_mean/pi1_mean)
            simple_conservative_performance_data.append(pi2_mean/pi1_mean)
            multimodal_performance_data.append(pi3_mean/pi1_mean)
            multimodal_conservative_performance_data.append(pi4_mean/pi1_mean)
            simple_variance_data.append(pi1_std/pi1_std)
            simple_conservative_variance_data.append(pi2_std/pi1_std)
            multimodal_variance_data.append(pi3_std/pi1_std)
            multimodal_conservative_variance_data.append(pi4_std/pi1_std)
        simple_performance_data = np.array(simple_performance_data)
        simple_conservative_performance_data = np.array(simple_conservative_performance_data)
        multimodal_performance_data = np.array(multimodal_performance_data)
        multimodal_conservative_performance_data = np.array(multimodal_conservative_performance_data)
        simple_variance_data = np.array(simple_variance_data)
        simple_conservative_variance_data = np.array(simple_conservative_variance_data)
        multimodal_variance_data = np.array(multimodal_variance_data)
        multimodal_conservative_variance_data = np.array(multimodal_conservative_variance_data)
        # track mean performance and variance of performance
        s_p_m = np.mean(simple_performance_data)
        sc_p_m = np.mean(simple_conservative_performance_data)
        m_p_m = np.mean(multimodal_performance_data)
        mc_p_m = np.mean(multimodal_conservative_performance_data)
        s_p_v = np.std(simple_performance_data)
        sc_p_v = np.std(simple_conservative_performance_data)
        m_p_v = np.std(multimodal_performance_data)
        mc_p_v = np.std(multimodal_conservative_performance_data)
        # track mean variance of score, and consistency of the variance
        s_v_m = np.mean(simple_variance_data)
        sc_v_m = np.mean(simple_conservative_variance_data)
        m_v_m = np.mean(multimodal_variance_data)
        mc_v_m = np.mean(multimodal_conservative_variance_data)
        s_v_v = np.std(simple_variance_data)
        sc_v_v = np.std(simple_conservative_variance_data)
        m_v_v = np.std(multimodal_variance_data)
        mc_v_v = np.std(multimodal_conservative_variance_data)
        print(s_p_m,sc_p_m,m_p_m)

        # store the data for that parameter setup
        simple_performances.append((s_p_m,s_p_v))
        simple_conservative_performances.append((sc_p_m,sc_p_v))
        multimodal_performances.append((m_p_m,m_p_v))
        multimodal_conservative_performances.append((mc_p_m,mc_p_v))
        simple_variances.append((s_v_m,s_v_v))
        simple_conservative_variances.append((sc_v_m,sc_v_v))
        multimodal_variances.append((m_v_m,m_v_v))
        multimodal_conservative_variances.append((mc_v_m,mc_v_v))
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
    ind = np.arange(len(runs))
    width = .15
    ax1.bar(ind,[i for i,j in simple_performances],width, yerr=[j for i,j in simple_performances], color='r', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax1.bar(ind + width,[i for i,j in simple_conservative_performances],width, yerr=[j for i,j in simple_conservative_performances], color='b', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax1.bar(ind + 2*width,[i for i,j in multimodal_performances],width, yerr=[j for i,j in multimodal_performances], color='g', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax1.bar(ind + 3*width,[i for i,j in multimodal_conservative_performances],width, yerr=[j for i,j in multimodal_conservative_performances], color='c', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax1.set_xlabel('upper_bound')
    ax1.set_title('Average cost of all trials')
    ax1.set_xticks(ind+2*width)
    ax1.set_xticklabels(runs)
    ax1.set_ylabel('performance against baseline')

    ax2.bar(ind,[i for i,j in simple_variances],width, yerr=[j for i,j in simple_variances], color='r', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax2.bar(ind + width,[i for i,j in simple_conservative_variances],width, yerr=[j for i,j in simple_conservative_variances], color='b', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax2.bar(ind + 2*width,[i for i,j in multimodal_variances],width, yerr=[j for i,j in multimodal_variances], color='g', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax2.bar(ind + 3*width,[i for i,j in multimodal_conservative_variances],width, yerr=[j for i,j in multimodal_conservative_variances], color='c', error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax2.set_xticks(ind+2*width)
    ax2.set_xticklabels(runs)
    ax2.set_xlabel('upper_bound')
    ax2.set_title('variance against baseline variance')

    modes = ['simple','simple_conservative','multimodal','multimodal_conservative']
    legend_lines = [Line2D([0], [0], color='r', lw=2),
                   Line2D([0], [0], color='b', lw=2),
                   Line2D([0], [0], color='g', lw=2),
                   Line2D([0], [0], color='c', lw=2),]
    plt.legend(legend_lines, modes)
    plt.title('Performance of different MDP representations vs upper_bound of edge cost of the graph')
    plt.show()
    graph.plt.close()

    pickle.dump(simple_performances,open('simple_p','w'))
    pickle.dump(simple_conservative_performances,open('simple_c_p.p','w'))
    pickle.dump(multimodal_performances,open('multimodal_p.p','w'))
    pickle.dump(multimodal_conservative_performances,open('multimodal_c_p.p','w'))
    pickle.dump(simple_variances,open('simple_v.p','w'))
    pickle.dump(simple_conservative_variances,open('simple_c_v.p','w'))
    pickle.dump(multimodal_variances,open('multimodal_v.p','w'))
    pickle.dump(multimodal_conservative_variances,open('multimodal_c_v.p','w'))
    pickle.dump(runs,open('runs.p','w'))

if __name__ == "__main__":
    run_main()
