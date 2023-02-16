import ot
import numpy as np
from typing import List


def histogram(A: List[float], bins: np.ndarray) -> np.ndarray:
    """
    Takes in a sample of scores, A, and then given the bins, creates a histogram. The following assumption must hold.

    Assumption: method assumes that bin[i] < bins[i+1] for all i < len(bins)
    also assumes that bin[0] <= all a in A <= bin[-1]

    :param A:  list of numbers (should be np.array or list type)
    :param bins: score bins, i.e. if scores are all integers 1-100 then the bins should
    be a list that looks like [1,2,...100].

    :return: histogram of scores snapped to the bins as vector
    """
    n = len(bins)
    hist = [0] * n
    for score in A:
        idx = np.abs(score - bins).argmin()
        hist[idx] += 1
    return np.array(hist)


def is_cdf(cdf: List[float], ep: float = 1e-4) -> bool:
    """
    A function to check whether the cdf that was provided is actually a cdf.

    :param cdf: a list containing the cdf values
    :param ep: the amount that the last cdf value should be within the number 1

    :return: a boolean that flags whether the cdf provided is actually a cdf
    """
    return (cdf[-1] - 1) < ep


def empirical_cdf(sample: np.ndarray, bins: np.ndarray = np.linspace(0, 1, 101)) -> np.ndarray:
    """
    Computes empirical cdf of sample, where sample are the scores for a specific group. If computing for tpr or fpr,
    then the sample is an array of scores of a specific group with a specific binary label, as defined by the metric.

    :param sample: scores for a specific group
    :param bins: score bins, i.e. if scores are all integers 1-100 then the bins should
    be a list that looks like [1,2,...100].

    :raises ValueError: Sample score distribution is not a valid cdf

    :return: returns a np.array X s.t. for each i < len(X), X[i] is the cdf value for
    the score corresponding to bin[i]
    """
    empiricalpdf = empirical_pdf(sample, bins)  # get pdf
    proposedcdf = cdf(empiricalpdf)  # cdf-ify the pdf

    if is_cdf(proposedcdf):
        proposedcdf = np.around(proposedcdf, 2)
        return proposedcdf
    else:
        raise ValueError("Sample score distribution is not a valid cdf")


def empirical_pdf(sample: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Computes empirical PDF of some sample, where sample are the scores for a specific group. If computing for tpr or
    fpr, then the sample is an array of scores of a specific group with a specific binary label, as defined by the
    metric.

    :param sample: scores for a specific group
    :param bins: score bins, i.e. if scores are all integers 1-100 then the bins should
    be a list that looks like [1,2,...100].

    :return: returns a np.array X s.t. for each i < len(X), X[i] is the PDF value for
    the score corresponding to bin[i] -- a normalized histogram
    """
    hist = histogram(sample, bins=bins)  # get histogram
    return hist / np.sum(hist)  # normalize histogram


def edm(vect: np.ndarray) -> np.ndarray:
    """
    Gets a Euclidean distance matrix for a vector

    :param vect: length n nd.array, assumes the vector is bin like, vect[i] < vect[i+1] for all i

    :return: a matrix where each M[i][j] is the distance between i,j
    """

    if len(vect.shape) > 1 and vect.shape[1] == 1:
        vect = vect.flatten()

    return np.abs(vect - np.expand_dims(vect, axis=1))


def cdf(A: np.ndarray) -> np.ndarray:
    """
    Turns a PDF into a cdf (this is just for readability)

    :param A: a PDF as an nd.array

    :return: the cdf of A
    """
    return np.cumsum(A, axis=0)


def psuedo_inverse(percentile: List[float], cdf: np.ndarray, bins: List[float]) -> float:
    """
    For some percentile, its pseudo inverse is the score which lives at that percentile
    This is not always well defined, hence why it is a psuedo inverse and not a true inverse
    see "Optimal Transport for Applied Mathematicians" page 54 def 2.1

    :param percentile: a number between 0,1
    :param cdf: some valid cdf as an np.array
    :param bins: distribution bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100].

    :return: the psuedo inverse of the percentile
    """

    cdf = np.insert(arr=np.around(cdf, 5), obj=0, values=0)
    inf = np.subtract(percentile, cdf)
    # See how far percentile is from the cdf percentiles
    # this will help us locate percentile in cdf nearest the percentile that is input
    # to the problem

    inf[inf <= 0] = np.Infinity
    inf_ix = inf.argmin()

    inverse = bins[inf_ix]

    return inverse


def barycenter(
        A: List[float],
        weights: List[float],
        bins: np.ndarray,
        reg: float = 1e-3,
        solver: str = "exact_LP") -> list:
    """
    Barycenter is computed using formula  in "Barycenters in the Wasserstein space" section 6.1

    :param A: list of samples of scores/numbers, should be list or nd.array. each list is the samples you wish to
    compute the barycenter of
    :param weights: the barycenter weights must sum to 1
    :param bins: distribution bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100].
    :param reg: regularizer parameter if the solver is NOT exact1D
    :param solver: possible inputs are
        - "anon"
        - "exact_1D"
        - "bregman"

    :raises ValueError: Sum of weights must add to 1
    :raises ValueError: This is not a supported solver

    :return: returns the empirical barycenter of A[0]...A[n]
    """
    if not np.abs(np.sum(weights)-1) < 1e-6:
        raise ValueError("Sum of weights must add to 1")

    if solver == "anon":
        cdfs = [empirical_cdf(sample, bins)
                for sample in A]  # Get the cdfs for each sample

        bc_pdf_all = []
        for i in range(len(A)):
            bc_i = np.zeros(shape=bins.shape)  # this is where the BC is stored
            # the source measure used for the pushforward
            mi_pdf = empirical_pdf(A[i], bins)
            for x in bins:
                # find ix of closest bin for some score
                bin_ix = np.abs(bins - x).argmin()
                Q_F_m0 = np.array(
                    [psuedo_inverse(cdfs[i][bin_ix], cdf, bins) for cdf in cdfs])
                # compute psuedo inverse across all of the input sample
                # compute new score using BC formula
                new_score = np.dot(Q_F_m0, weights)

                newscore_bin_ix = np.abs(bins - new_score).argmin()
                # find ix of the bin of the new score

                bc_i[newscore_bin_ix] += mi_pdf[bin_ix]
                # give the new score the probability of the source measure
            bc_pdf_all.append(bc_i)

        bc_pdf = np.array(bc_pdf_all).mean(axis=0)
    elif solver == "exact_LP":
        weights = np.array(weights)
        A = np.vstack([empirical_pdf(a, bins) for a in A]).T

        M = edm(bins)
        M = np.divide(M, M.max())
        bc_pdf = ot.lp.barycenter(
            A, M, weights, solver='interior-point', verbose=True)
    elif solver == "bregman":
        weights = np.array(weights)
        A = np.vstack([empirical_pdf(a, bins) for a in A]).T
        M = edm(bins)
        M = np.divide(M, M.max())
        bc_pdf = ot.bregman.barycenter(A, M, reg, weights)
    else:
        raise ValueError("This is not a supported solver.")
    return bc_pdf


def transport(x: float, src: List[float], dst: List[float], bins: np.ndarray) -> float:
    """
    Performs the transport from the source distribution to the destination distribution via the barycenter.

    :param x: the score to be transported, some val s.t. bin[0] <= x <= bin[-1]
    :param src: the source distribution in the transport, given as a list of values
    :param dst: the destination distribution, assumed to be a barycenter and given as a pdf
    :param bins: distribution bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100].

    :return: the transported value
    """

    src_cdf = empirical_cdf(src, bins)
    dst_cdf = cdf(dst)
    bin_ix = np.abs(bins - x).argmin()
    q = src_cdf[bin_ix]
    transported_score = psuedo_inverse(q, dst_cdf, bins=bins)

    return transported_score
