# coding=utf-8
"""
Fingerprint matching code for audfprint
2014-05-26 Dan Ellis dpwe@ee.columbia.edu
"""
from __future__ import division, print_function

from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from afp.audfprint.hash_table import HashTable
from afp.audfprint.peak_extractor import Audfprint_peaks


def encpowerof2(val: float) -> int:
    """
    Return N s.t. 2^N >= val.
    """
    return int(np.ceil(np.log(max(1, val)) / np.log(2)))


def locmax(vec: npt.NDArray[Any], indices: bool = False) -> npt.NDArray[Any]:
    """
    Return a boolean vector of which points in vec are local maxima.
    End points are peaks if larger than single neighbors.
    if indices=True, return the indices of the True values instead
    of the boolean vector. (originally from audfprint.py)
    """
    # x[-1]-1 means last value can be a peak
    # nbr = np.greater_equal(np.r_[x, x[-1]-1], np.r_[x[0], x])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(vec) + 1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = nbr[:-1] & ~nbr[1:]
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask


def keep_local_maxes(vec: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """
    Zero out values unless they are local maxima.
    """
    local_maxes = np.zeros(vec.shape, dtype=np.float32)
    locmaxindices = locmax(vec, indices=True)
    local_maxes[locmaxindices] = vec[locmaxindices]
    return local_maxes


def find_modes(
    data: npt.NDArray[np.float32], threshold: int = 5
) -> Tuple[npt.NDArray[Any], npt.NDArray[np.float32]]:
    """
    Find multiple modes in data,  Report a list of (mode, count) pairs for every mode greater
    than or equal to threshold. Only local maxima in counts are returned.
    """
    # Ignores window at present
    datamin = np.amin(data)
    fullvector = np.bincount(data - datamin)
    # Find local maximas
    localmaxes = np.nonzero(
        np.logical_and(locmax(fullvector), np.greater_equal(fullvector, threshold))
    )[0]
    return localmaxes + datamin, fullvector[localmaxes]


class Matcher(object):
    """
    Provide matching for audfprint fingerprint queries to hash table.
    """

    def __init__(self) -> None:
        """
        Set up default object values.
        """
        # Tolerance window for time differences
        self.window = 2
        # Absolute minimum number of matching hashes to count as a match
        self.threshcount = 5
        # How many hits to return?
        self.max_returns = 1
        # How deep to search in return list?
        self.search_depth = 100
        # Sort those returns by time (instead of counts)?
        self.sort_by_time = False
        # Verbose reporting?
        self.verbose = 1
        # Careful counts?
        self.exact_count = False
        # Search for time range?
        self.find_time_range = False
        # Quantile of time range to report.
        self.time_quantile = 0.05
        # If there are a lot of matches within a single track at different
        # alignments, stop looking after a while.
        self.max_alignments_per_id = 100

    def _best_count_ids(
        self, hits: npt.NDArray[np.float32], ht: HashTable
    ) -> Tuple[Any, Any]:
        """
        Return the indexes for the ids with the best counts.
        hits is a matrix as returned by hash_table.get_hits()
        with rows of consisting of [id dtime hash otime].
        """
        allids = hits[:, 0]
        ids = np.unique(allids)
        # rawcounts = np.sum(np.equal.outer(ids, allids), axis=1)
        # much faster, and doesn't explode memory
        rawcounts = np.bincount(allids)[ids]
        # Divide the raw counts by the total number of hashes stored
        # for the ref track, to downweight large numbers of chance
        # matches against longer reference tracks.
        wtdcounts = rawcounts / (ht.hashesperid[ids].astype(float))

        # Find all the actual hits for a the most popular ids
        bestcountsixs = np.argsort(wtdcounts)[::-1]
        # We will examine however many hits have rawcounts above threshold
        # up to a maximum of search_depth.
        maxdepth = np.minimum(
            np.count_nonzero(np.greater(rawcounts, self.threshcount)), self.search_depth
        )
        # Return the ids to check
        bestcountsixs = bestcountsixs[:maxdepth]
        return ids[bestcountsixs], rawcounts[bestcountsixs]

    def _unique_match_hashes(self, id: int, hits: npt.NDArray[Any], mode: float) -> Any:
        """
        Return the list of unique matching hashes.  Split out so
        we can recover the actual matching hashes for the best
        match if required.
        """
        allids = hits[:, 0]
        alltimes = hits[:, 1]
        allhashes = hits[:, 2].astype(np.int64)
        allotimes = hits[:, 3]
        timebits = max(1, encpowerof2(np.amax(allotimes)))

        matchix = np.nonzero(
            np.logical_and(
                allids == id, np.less_equal(np.abs(alltimes - mode), self.window)
            )
        )[0]
        matchhasheshash = np.unique(
            allotimes[matchix] + (allhashes[matchix] << timebits)
        )
        timemask = (1 << timebits) - 1
        matchhashes = np.c_[matchhasheshash & timemask, matchhasheshash >> timebits]
        return matchhashes

    def _calculate_time_ranges(
        self, hits: npt.NDArray[Any], id: int, mode: float
    ) -> Tuple[int, int]:
        """
        Given the id and mode, return the actual time support.
        hits is an np.array of id, skew_time, hash, orig_time
        which must be sorted in orig_time order.
        """
        minoffset = mode - self.window
        maxoffset = mode + self.window
        # match_times = sorted(hits[row, 3]
        #                     for row in np.nonzero(hits[:, 0]==id)[0]
        #                     if mode - self.window <= hits[row, 1]
        #                     and hits[row, 1] <= mode + self.window)
        match_times = hits[
            np.logical_and.reduce(
                [hits[:, 1] >= minoffset, hits[:, 1] <= maxoffset, hits[:, 0] == id]
            ),
            3,
        ]
        min_time = match_times[int(len(match_times) * self.time_quantile)]
        max_time = match_times[int(len(match_times) * (1.0 - self.time_quantile)) - 1]
        # log("_calc_time_ranges: len(hits)={:d} id={:d} mode={:d} matches={:d} min={:d} max={:d}".format(
        #    len(hits), id, mode, np.sum(np.logical_and(hits[:, 1] >= minoffset,
        #                                               hits[:, 1] <= maxoffset)),
        #    min_time, max_time))
        return min_time, max_time

    def _exact_match_counts(
        self, hits: npt.NDArray[Any], ids: List[int], rawcounts: List[int]
    ) -> npt.NDArray[np.int32]:
        """
        Find the number of "filtered" (time-consistent) matching hashes for each of the promising ids in <ids>.
        Return an np.array whose rows are [id, filtered_count, modal_time_skew, unfiltered_count, original_rank,
        min_time, max_time].  Results are sorted by original rank (but will not in general include all the the original IDs).
        There can be multiple rows for a single ID, if there are several distinct time_skews giving good matches.
        """
        # Sort hits into time_in_original order - needed for _calc_time_range
        sorted_hits = hits[hits[:, 3].argsort()]
        # Slower, old process for exact match counts
        allids = sorted_hits[:, 0]
        alltimes = sorted_hits[:, 1]
        # allhashes = sorted_hits[:, 2]
        # allotimes = sorted_hits[:, 3]
        # Allocate enough space initially for 4 modes per hit
        maxnresults = len(ids) * 4
        results = np.zeros((maxnresults, 7), np.int32)
        nresults = 0
        min_time = 0
        max_time = 0
        for urank, (id, rawcount) in enumerate(zip(ids, rawcounts)):
            modes, _ = find_modes(
                alltimes[np.nonzero(allids == id)[0]],
                threshold=self.threshcount,
            )
            for mode in modes:
                matchhashes = self._unique_match_hashes(id, sorted_hits, mode)
                # Now we get the exact count
                filtcount = len(matchhashes)
                if filtcount >= self.threshcount:
                    if nresults == maxnresults:
                        # Extend array
                        maxnresults *= 2
                        results.resize((maxnresults, results.shape[1]))
                    if self.find_time_range:
                        min_time, max_time = self._calculate_time_ranges(
                            sorted_hits, id, mode
                        )
                    results[nresults, :] = [
                        id,
                        filtcount,
                        mode,
                        rawcount,
                        urank,
                        min_time,
                        max_time,
                    ]
                    nresults += 1
        return results[:nresults, :]

    def _approx_match_counts(
        self, hits: npt.NDArray[np.float32], ids: List[int], rawcounts: List[int]
    ) -> npt.NDArray[np.int32]:
        """
        Quick and slightly inaccurate routine to count time-aligned hits. Only considers largest mode for reference ID match.
        ---
        Args:
            hits: np.array of
                Hash matches, each row consists of <track_id, skew_time, hash, orig_time>.
            ids: list of the
                IDs to check, based on raw match count.
            rawcounts: list giving the
                Actual raw counts for each id to try.
        Returns:
            npt.NDArray[np.int32]:
                Rows of [id, filt_count, time_skew, raw_count, orig_rank, min_time, max_time].
                Ids occur in the same order as the input list, but ordering of (potentially multiple)
                hits within each track may not be sorted (they are sorted by the largest single count value, not
                the total count integrated over -window:+window bins).
        """
        # In fact, the counts should be the same as exact_match_counts
        # *but* some matches may be pruned because we don't bother to
        # apply the window (allowable drift in time alignment) unless
        # there are more than threshcount matches at the single best time skew.
        # Note: now we allow multiple matches per ID, this may need to grow
        # so it can grow inside the loop.
        results = np.zeros((len(ids), 7), np.int32)
        if not hits.size:
            # No hits found, return empty results
            return results
        # Sort hits into time_in_original order - needed for _calc_time_range
        sorted_hits = hits[hits[:, 3].argsort()]
        allids = sorted_hits[:, 0].astype(int)
        alltimes = sorted_hits[:, 1].astype(int)
        # Make sure every value in alltimes is >=0 for bincount
        mintime = np.amin(alltimes)
        alltimes -= mintime
        nresults = 0
        min_time = 0
        max_time = 0
        for urank, (id, rawcount) in enumerate(zip(ids, rawcounts)):
            # Make sure id is an int64 before shifting it up.
            id = int(id)
            # Select the subrange of bincounts corresponding to this id
            bincounts = np.bincount(alltimes[allids == id])
            still_looking = True
            # Only consider legit local maxima in bincounts.
            filtered_bincounts = keep_local_maxes(bincounts)
            found_this_id = 0
            while still_looking:
                mode = np.argmax(filtered_bincounts)
                if filtered_bincounts[mode] <= self.threshcount:
                    # Too few - skip to the next id
                    still_looking = False
                    continue
                count = np.sum(
                    bincounts[max(0, mode - self.window) : (mode + self.window + 1)]
                )
                if self.find_time_range:
                    min_time, max_time = self._calculate_time_ranges(
                        sorted_hits, id, mode + mintime
                    )
                results[nresults, :] = [
                    id,
                    count,
                    mode + mintime,
                    rawcount,
                    urank,
                    min_time,
                    max_time,
                ]
                nresults += 1
                if nresults >= results.shape[0]:
                    results = np.vstack([results, np.zeros(results.shape, np.int32)])
                # Clear this hit to find next largest.
                filtered_bincounts[
                    max(0, mode - self.window) : (mode + self.window + 1)
                ] = 0
                found_this_id += 1
                if found_this_id > self.max_alignments_per_id:
                    still_looking = False
        return results[:nresults, :]

    def match_hashes(
        self,
        ht: HashTable,
        hashes: npt.NDArray[np.float32],
        hashesfor: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """
        Match audio against fingerprint hash table. Return top N matches as (id, filteredmatches, timoffs, rawmatches,
        origrank, mintime, maxtime). If hashesfor specified, return the actual matching hashes for that hit (0=top hit).
        """
        # find the implicated id, time pairs from hash table
        # log("nhashes=%d" % np.shape(hashes)[0])
        hits = ht.get_hits(hashes)
        bestids, rawcounts = self._best_count_ids(hits, ht)

        # log("len(rawcounts)=%d max(rawcounts)=%d" %
        #    (len(rawcounts), max(rawcounts)))
        if not self.exact_count:
            results = self._approx_match_counts(hits, bestids, rawcounts)
        else:
            results = self._exact_match_counts(hits, bestids, rawcounts)
        # Sort results by filtered count, descending

        results = results[(-results[:, 1]).argsort(),]

        if hashesfor is None:
            return results, None
        else:
            id = results[hashesfor, 0]
            mode = results[hashesfor, 2]
            hashesforhashes = self._unique_match_hashes(id, hits, mode)
            return results, hashesforhashes

    def match_file(
        self, analyzer: Audfprint_peaks, ht: HashTable, filename: str
    ) -> Tuple[Any, float, int]:
        """
        Read in an audio file, calculate its landmarks, query against hash table.
        Return top N matches as (id, filterdmatchcount, timeoffs, rawmatchcount),
        also length of input file in sec, and count of raw query hashes extracted.
        """
        q_hashes = analyzer.wavfile2hashes(filename)

        # Fake durations as largest hash time
        if len(q_hashes) == 0:
            durd = 0.0
        else:
            durd = analyzer.n_hop * q_hashes[-1][0] / analyzer.target_sr
        # Run query
        rslts, _ = self.match_hashes(ht, q_hashes)
        # Post filtering
        if self.sort_by_time:
            rslts = rslts[(-rslts[:, 2]).argsort(), :]
        return rslts[: self.max_returns, :], durd, len(q_hashes)

    def file_match_to_msgs(
        self,
        analyzer: Audfprint_peaks,
        ht: HashTable,
        qry: str,
    ) -> Tuple[str, str, int]:
        """
        Perform a match on a single input file, return list of message strings.
        """
        rslts, dur, nhash = self.match_file(analyzer, ht, qry)
        t_hop = analyzer.n_hop / analyzer.target_sr

        if self.verbose:
            qrymsg = qry + (" %.1f " % dur) + "sec " + str(nhash) + " raw hashes"
        else:
            qrymsg = qry

        msgrslt = []
        if len(rslts) == 0:
            # No matches returned at all
            nhashaligned = 0
            if self.verbose:
                msgrslt.append("NOMATCH " + qrymsg)
            else:
                msgrslt.append(qrymsg + "\t")

            return "NOMATCH", "", 0

        else:
            for (
                tophitid,
                nhashaligned,
                aligntime,
                nhashraw,
                rank,
                min_time,
                max_time,
            ) in rslts:
                # figure the number of raw and aligned matches for top hit
                if self.verbose:
                    if self.find_time_range:
                        msg = (
                            "Matched {:6.1f} s starting at {:6.1f} s in {:s}"
                            " to time {:6.1f} s in {:s}"
                        ).format(
                            (max_time - min_time) * t_hop,
                            min_time * t_hop,
                            qry,
                            (min_time + aligntime) * t_hop,
                            ht.names[tophitid],
                        )
                    else:
                        msg = "Matched {:s} as {:s} at {:6.1f} s".format(
                            qrymsg, ht.names[tophitid], aligntime * t_hop
                        )
                    msg += (
                        " with {:5d} of {:5d} common hashes" " at rank {:2d}"
                    ).format(nhashaligned, nhashraw, rank)
                    msgrslt.append(msg)
                else:
                    msgrslt.append(qrymsg + "\t" + ht.names[tophitid])

            return "MATCH", ht.names[tophitid], nhashaligned
