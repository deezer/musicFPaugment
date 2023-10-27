# coding=utf-8
"""
Python implementation of the very simple, fixed-array hash table used for the audfprint fingerprinter.
2014-05-25 Dan Ellis dpwe@ee.columbia.edu
"""
from __future__ import division, print_function

import gzip
import math
import pickle  # Py3
import random
from typing import Any, Callable, List, Optional, Union

import numpy as np
import numpy.typing as npt

basestring = (str, bytes)  # Py3


# Current format version
HT_VERSION = 20170724
# Earliest acceptable version
HT_COMPAT_VERSION = 20170724
# Earliest version that can be updated with load_old
HT_OLD_COMPAT_VERSION = 20140920


def _bitsfor(maxval: int) -> int:
    """
    Convert a maxval into a number of bits (left shift). Raises a ValueError if the maxval is not a power of 2.
    """
    maxvalbits = int(round(math.log(maxval) / math.log(2)))
    if maxval != (1 << maxvalbits):
        raise ValueError("maxval must be a power of 2, not %d" % maxval)
    return maxvalbits


class HashTable(object):
    """
    Simple hash table for storing and retrieving fingerprint hashes.
       >>> ht = HashTable(size=2**10, depth=100)
       >>> ht.store('identifier', list_of_landmark_time_hash_pairs)
       >>> list_of_ids_tracks = ht.get_hits(hash)
    """

    def __init__(self, filename: Optional[str] = None):
        """
        Allocate an empty hash table of the specified size.
        """
        if filename is not None:
            self.load(filename)
        else:
            self.hashbits = 20
            self.depth = 100
            self.maxtimebits = _bitsfor(16384)
            # allocate the big table
            size = 2**self.hashbits
            self.table = np.zeros((size, self.depth), dtype=np.uint32)
            # keep track of number of entries in each list
            self.counts = np.zeros(size, dtype=np.int32)
            # map names to IDs
            self.names: List[Any] = []
            # track number of hashes stored per id
            self.hashesperid = np.zeros(0, np.uint32)
            # Record the current version
            self.ht_version = HT_VERSION
            # Mark as unsaved
            self.dirty = True

    def store(
        self,
        name: Union[int, str],
        timehashpairs: npt.NDArray[np.int32],
    ) -> None:
        """
        Store a list of hashes in the hash table associated with a particular name (or integer ID) and time.
        """
        id_ = self.name_to_id(name, add_if_missing=True)
        # Now insert the hashes
        hashmask = (1 << self.hashbits) - 1

        maxtime = 1 << self.maxtimebits

        timemask = maxtime - 1

        # Try sorting the pairs by hash value, for better locality in storing
        # sortedpairs = sorted(timehashpairs, key=lambda x:x[1])
        # sortedpairs = timehashpairs
        idval = (id_ + 1) << self.maxtimebits

        for time_, hash_ in timehashpairs:  # sortedpairs:
            # Keep only the bottom part of the hash value
            hash_ &= hashmask
            # How many already stored for this hash?
            count = self.counts[hash_]
            # Keep only the bottom part of the time value
            # time_ %= mxtime
            time_ &= timemask
            # Mixin with ID
            val = idval + time_  # .astype(np.uint32)
            if count < self.depth:
                # insert new val in next empty slot
                # slot = self.counts[hash_]
                self.table[hash_, count] = val
            else:
                # Choose a point at random
                slot = random.randint(0, count)
                # Only store if random slot wasn't beyond end
                if slot < self.depth:
                    self.table[hash_, slot] = val
            # Update record of number of vals in this bucket
            self.counts[hash_] = count + 1
        # Record how many hashes we (attempted to) save for this id
        self.hashesperid[id_] += len(timehashpairs)
        # Mark as unsaved
        self.dirty = True

    def save(self, name: str) -> None:
        """
        Save hash table to file name, including optional addition params
        """

        f = gzip.open(name, "wb")

        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.dirty = False
        nhashes = sum(self.counts)
        # Report the proportion of dropped hashes (overfull table)

        dropped = nhashes - sum(np.minimum(self.depth, self.counts))
        print(
            "Saved fprints for",
            sum(n is not None for n in self.names),
            "files (",
            nhashes,
            "hashes) to",
            name,
            "(%.2f%% dropped)" % (100.0 * dropped / max(1, nhashes)),
        )

    def load(self, name: str) -> None:
        """
        Read either pklz or mat-format hash table file.
        """
        self.load_pkl(name)
        nhashes = sum(self.counts)
        # Report the proportion of dropped hashes (overfull table)
        dropped = nhashes - sum(np.minimum(self.depth, self.counts))

        print(
            "Read fprints for",
            sum(n is not None for n in self.names),
            "files (",
            nhashes,
            "hashes) from",
            name,
            "(%.2f%% dropped)" % (100.0 * dropped / max(1, nhashes)),
        )

    def load_pkl(self, name: str, file_object: Any = None) -> None:
        """
        Read hash table values from pickle file name.
        """
        if file_object:
            f = file_object
        else:
            f = gzip.open(name, "rb")
        temp = pickle.load(f, encoding="latin1")
        if temp.ht_version < HT_OLD_COMPAT_VERSION:
            raise ValueError(
                "Version of "
                + name
                + " is "
                + str(temp.ht_version)
                + " which is not at least "
                + str(HT_OLD_COMPAT_VERSION)
            )
        # assert temp.ht_version >= HT_COMPAT_VERSION
        self.hashbits = temp.hashbits
        self.depth = temp.depth
        if hasattr(temp, "maxtimebits"):
            self.maxtimebits = temp.maxtimebits
        else:
            self.maxtimebits = _bitsfor(temp.maxtime)
        if temp.ht_version < HT_COMPAT_VERSION:
            # Need to upgrade the database.
            print("Loading database version", temp.ht_version, "in compatibility mode.")
            # Offset all the nonzero bins with one ID count.
            temp.table += np.array(1 << self.maxtimebits).astype(np.uint32) * (
                temp.table != 0
            )
            temp.ht_version = HT_VERSION
        self.table = temp.table
        self.ht_version = temp.ht_version
        self.counts = temp.counts
        self.names = temp.names
        self.hashesperid = np.array(temp.hashesperid).astype(np.uint32)
        self.dirty = False

    def reset(self) -> None:
        """
        Reset to empty state (but preserve parameters).
        """
        self.table[:, :] = 0
        self.counts[:] = 0
        self.names = []
        self.hashesperid.resize(0)
        self.dirty = True

    def get_entry(self, hash_: int) -> Any:
        """
        Return np.array of [id, time] entries associate with the given hash as rows.
        """
        vals = self.table[hash_, : min(self.depth, self.counts[hash_])]
        maxtimemask = (1 << self.maxtimebits) - 1
        # ids we report externally start at 0, but in table they start at 1.
        ids = (vals >> self.maxtimebits) - 1
        return np.c_[ids, vals & maxtimemask].astype(np.int32)

    def get_hits(self, hashes: npt.NDArray[np.float32]) -> npt.NDArray[Any]:
        """
        Return np.array of [id, delta_time, hash, time] rows associated with each element
        in hashes array of [time, hash] rows. This version has get_entry() inlined, it's about 30% faster.
        """
        # Allocate to largest possible number of hits
        nhashes = np.shape(hashes)[0]
        hits = np.zeros((nhashes * self.depth, 4), np.int32)
        nhits = 0
        maxtimemask = (1 << self.maxtimebits) - 1
        hashmask = (1 << self.hashbits) - 1
        # Fill in
        for ix in range(nhashes):
            time_ = hashes[ix][0]
            hash_ = hashmask & hashes[ix][1]
            nids = min(self.depth, self.counts[hash_])
            tabvals = self.table[hash_, :nids]
            hitrows = nhits + np.arange(nids)
            # Make external IDs start from 0.
            hits[hitrows, 0] = (tabvals >> self.maxtimebits) - 1
            hits[hitrows, 1] = (tabvals & maxtimemask) - time_
            hits[hitrows, 2] = hash_
            hits[hitrows, 3] = time_
            nhits += nids
        # Discard the excess rows
        hits.resize((nhits, 4))
        return hits

    def totalhashes(self) -> np.int32:
        """
        Return the total count of hashes stored in the table.
        """
        return np.sum(self.counts)

    def name_to_id(self, name: Union[int, str], add_if_missing: bool = False) -> int:
        """
        Lookup name in the names list, or optionally add.
        """
        if isinstance(name, basestring):
            # lookup name or assign new
            if name not in self.names:
                if not add_if_missing:
                    raise ValueError("name " + name + " not found")
                # Use an empty slot in the list if one exists.
                try:
                    id_ = self.names.index(None)
                    self.names[id_] = name
                    self.hashesperid[id_] = 0
                except ValueError:
                    self.names.append(name)
                    self.hashesperid = np.append(self.hashesperid, [0])
            id_ = self.names.index(name)
        else:
            # we were passed in a numerical id
            id_ = name
        return id_

    def remove(self, name: Union[str, int]) -> None:
        """
        Remove all data for named entity from the hash table.
        """
        id_ = self.name_to_id(name)
        # Top nybbles of table entries are id_ + 1 (to avoid all-zero entries)
        id_in_table = (self.table >> self.maxtimebits) == id_ + 1
        hashes_removed = 0
        for hash_ in np.nonzero(np.max(id_in_table, axis=1))[0]:
            table_vals = self.table[hash_, : self.counts[hash_]]
            vals = [v for v, x in zip(table_vals, id_in_table[hash_]) if not x]
            self.table[hash_] = np.hstack([vals, np.zeros(self.depth - len(vals))])
            # This will forget how many extra hashes we had dropped until now.
            self.counts[hash_] = len(vals)
            hashes_removed += np.sum(id_in_table[hash_])
        self.names[id_] = None
        self.hashesperid[id_] = 0
        self.dirty = True
        print("Removed", name, "(", hashes_removed, "hashes).")

    def retrieve(self, name: Union[str, int]) -> npt.NDArray[np.int32]:
        """
        Return an np.array of (time, hash) pairs found in the table.
        """
        id_ = self.name_to_id(name)
        maxtimemask = (1 << self.maxtimebits) - 1
        num_hashes_per_hash = np.sum(
            (self.table >> self.maxtimebits) == (id_ + 1), axis=1
        )
        hashes_containing_id = np.nonzero(num_hashes_per_hash)[0]
        timehashpairs = np.zeros((sum(num_hashes_per_hash), 2), dtype=np.int32)
        hashes_so_far = 0
        for hash_ in hashes_containing_id:
            entries = self.table[hash_, : self.counts[hash_]]
            matching_entries = np.nonzero((entries >> self.maxtimebits) == (id_ + 1))[0]
            times = entries[matching_entries] & maxtimemask
            timehashpairs[hashes_so_far : hashes_so_far + len(times), 0] = times
            timehashpairs[hashes_so_far : hashes_so_far + len(times), 1] = hash_
            hashes_so_far += len(times)
        return timehashpairs

    def list(self, print_fn: Optional[Callable[[str], None]] = None) -> None:
        """
        List all the known items.
        """
        if not print_fn:
            print_fn = print
        for name, count in zip(self.names, self.hashesperid):
            if name:
                print_fn(name + " (" + str(count) + " hashes)")
