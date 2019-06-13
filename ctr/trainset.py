'''This module contains the Trainset class.'''


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from six import iteritems


class Trainset:

    def __init__(self, ur, ir, n_users, n_items, n_ratings,
                 raw2inner_id_users, raw2inner_id_items):

        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None
        # inner2raw dicts could be built right now (or even before) but they
        # are not always useful so we wait until we need them.
        self._inner2raw_id_users = None
        self._inner2raw_id_items = None

    def knows_user(self, uid):
        """Indicate if the user is part of the trainset.

        A user is part of the trainset if the user has at least one rating.

        Args:
            uid(int): The (inner) user id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if user is part of the trainset, else ``False``.
        """

        return uid in self.ur

    def knows_item(self, iid):
        """Indicate if the item is part of the trainset.

        An item is part of the trainset if the item was rated at least once.

        Args:
            iid(int): The (inner) item id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if item is part of the trainset, else ``False``.
        """

        return iid in self.ir

    def to_inner_uid(self, ruid):
        """Convert a **user** raw id to an inner id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            ruid(str): The user raw id.

        Returns:
            int: The user inner id.

        Raises:
            ValueError: When user is not part of the trainset.
        """

        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError('User ' + str(ruid) +
                             ' is not part of the trainset.')

    def to_raw_uid(self, iuid):
        """Convert a **user** inner id to a raw id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            iuid(int): The user inner id.

        Returns:
            str: The user raw id.

        Raises:
            ValueError: When ``iuid`` is not an inner id.
        """

        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_users)}

        try:
            return self._inner2raw_id_users[iuid]
        except KeyError:
            raise ValueError(str(iuid) + ' is not a valid inner id.')

    def to_inner_iid(self, riid):
        """Convert an **item** raw id to an inner id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            riid: The item raw id.

        Returns:
            int: The item inner id.

        Raises:
            ValueError: When item is not part of the trainset.
        """

        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError('Item ' + str(riid) +
                             ' is not part of the trainset.')

    def to_raw_iid(self, iiid):
        """Convert an **item** inner id to a raw id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            iiid(int): The item inner id.

        Returns:
            str: The item raw id.

        Raises:
            ValueError: When ``iiid`` is not an inner id.
        """

        if self._inner2raw_id_items is None:
            self._inner2raw_id_items = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_items)}

        try:
            return self._inner2raw_id_items[iiid]
        except KeyError:
            raise ValueError(str(iiid) + ' is not a valid inner id.')

    def all_ratings(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids (see
            :ref:`this note <raw_inner_note>`).
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

    def build_testset(self):
        return [(self.to_raw_uid(u), self.to_raw_iid(i), r)
                for (u, i, r) in self.all_ratings()]

    def build_anti_testset(self, fill=None):
        """
        The ratings are all the ratings that are **not** in the trainset, i.e.
        all the ratings :math:`r_{ui}` where the user :math:`u` is known, the
        item :math:`i` is known, but the rating :math:`r_{ui}`  is not in the
        trainset. As :math:`r_{ui}` is unknown, it is either replaced by the
        :code:`fill` value or assumed to be equal to the mean of all ratings
        :meth:`global_mean <surprise.Trainset.global_mean>`.

        Args:
            fill(float): The value to fill unknown ratings. If :code:`None` the
                global mean of all ratings :meth:`global_mean
                <surprise.Trainset.global_mean>` will be used.

        Returns:
            A list of tuples ``(uid, iid, fill)`` where ids are raw ids.
        """
        fill = self.global_mean if fill is None else float(fill)

        anti_testset = []
        for u in self.all_users():
            user_items = set([j for (j, _) in self.ur[u]])
            anti_testset += [(self.to_raw_uid(u), self.to_raw_iid(i), fill) for
                             i in self.all_items() if
                             i not in user_items]
        return anti_testset

    def all_users(self):
        """Generator function to iterate over all users.

        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_raw_users(self):
        return list(self._raw2inner_id_users.keys())

    def all_raw_items(self):
        return list(self._raw2inner_id_items.keys())

    def all_items(self):
        """Generator function to iterate over all items.

        Yields:
            Inner id of items.
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean
