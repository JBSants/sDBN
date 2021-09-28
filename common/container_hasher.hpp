#ifndef CONTAINER_HASH_H
#define CONTAINER_HASH_H

#ifdef BOOST_FOUND
#include <boost/container_hash/hash.hpp>
#endif

template <typename Container> // we can make this generic for any container [1]
struct ContainerHash {
    std::size_t operator()(Container const& c) const {
        #ifdef BOOST_FOUND
            return boost::hash_range(c.begin(), c.end());
        #else
            int hash = c.size();

            for(auto &i : c) {
                hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }

            return hash;
        #endif

    }
};

#endif