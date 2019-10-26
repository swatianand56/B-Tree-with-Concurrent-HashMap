package cs764;


import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.lang3.tuple.ImmutablePair;


/**
 * Any search/insert call on this CHM will acquire a read lock, to ensure only one thread gets to
 * convert the CHM into a B-tree node.
 *
 * Duplicate keys are not allowed.
 */
public class ConcurrentMap<K extends Comparable, V> {

//  private final ReadLock readLock;
//  private final WriteLock writeLock;
  private ConcurrentHashMap<K, V> map;
//  private K lastMax;

  public ConcurrentMap(int maxSize, int numThreads) {
    this.map = new ConcurrentHashMap<>(maxSize + numThreads * 2, 0.75f, numThreads);
//    ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
//    this.readLock = lock.readLock();
//    this.writeLock = lock.writeLock();
//    lastMax = null;/
  }

  /**
   * The invoking thread should keep retrying until the insert succeeds.
   *
   * @throws RuntimeException if duplicates are inserted
   */
  void insertTuple(K key, V value) {
//    this.readLock.lock();

//    if (lastMax != null && lastMax.compareTo(key) < 0) {
//      // retry insert, has to go into B-tree
//      this.readLock.unlock();
//      throw new cs764.ConversionException("Retry!");
//    }

    if (this.map.containsKey(key)) {
//      this.readLock.unlock();
      throw new RuntimeException("Duplicates not allowed");
    }

    this.map.put(key, value);
//    this.readLock.unlock();
  }

  // convert to B-tree node

  // if(size > blah) {map.exportAndClear}
  public ImmutablePair<List<K>, List<V>> exportAndClear() {
//    this.writeLock.lock();

    List<K> keys = new ArrayList<>();
    List<V> values = new ArrayList<>();

    Map<K, V> treemap = new TreeMap<>(this.map);

    for (Map.Entry<K, V> entry : treemap.entrySet()) {
      keys.add(entry.getKey());
      values.add(entry.getValue());
    }

    this.map.clear();
//    this.lastMax = keys.get(keys.size() - 1);

//    this.writeLock.unlock();

    return ImmutablePair.of(keys, values);
  }

  int size() {
    return this.map.size();
  }

  /**
   *
   */
  public V search(K key) {
//    this.readLock.lock();

//    if (!this.readLock.tryLock()) {
//      // todo: come up with a better exception name
//      throw new RuntimeException("Conversion in progress, you may not search now, please retry");
//    }

    //    this.readLock.unlock();

    return this.map.get(key);
  }

  // range search - todo: is it even required?
}
