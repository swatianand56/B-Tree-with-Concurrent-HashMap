package cs764;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.apache.commons.lang3.tuple.ImmutablePair;


public class BPlusTree<K extends Comparable<? super K>, V> {

  // use fair-policy (arrival order policy)
  private final ReentrantReadWriteLock rwl = new ReentrantReadWriteLock(true);
  // for all other operations except CHM conversion
  private final Lock readLock = rwl.readLock();
  // only while converting the CHM we use writeLock
  private final Lock writeLock = rwl.writeLock();

  private K maxKeyInBTree = null;
  private final int maxCHMSize;

  private ConcurrentMap<K, V> chm;

  public enum RangePolicy {
    EXCLUSIVE, INCLUSIVE
  }

  /**
   * The branching factor used when none specified in constructor.
   */
  private static final int DEFAULT_BRANCHING_FACTOR = 128;

  private static final int DEFAULT_CHM_SIZE = (int) (DEFAULT_BRANCHING_FACTOR * 0.75);

  private static final double DEFAULT_CHM_CONCURRENCY_FACTOR = 0.75;

  /**
   * The branching factor for the B+ tree, that measures the capacity of nodes (i.e., the number of
   * children nodes) for internal nodes in the tree.
   */
  private int branchingFactor;

  /**
   * The root node of the B+ tree.
   */
  private Node root;

  private BPlusTree(int numThreads) {
    this(DEFAULT_BRANCHING_FACTOR, DEFAULT_CHM_SIZE, numThreads, DEFAULT_CHM_CONCURRENCY_FACTOR);
  }

  private BPlusTree(int branchingFactor, int maxCHMSize, int numThreads, double chmConcurrencyFactor) {
    if (branchingFactor <= 2) {
      throw new IllegalArgumentException("Illegal branching factor: "
          + branchingFactor);}

    this.branchingFactor = branchingFactor;
    this.maxCHMSize = maxCHMSize;
    root = new LeafNode();

    this.chm = new ConcurrentMap<>(maxCHMSize, (int) Math.ceil(numThreads*chmConcurrencyFactor));
  }




  /**
   * Returns the value to which the specified key is associated, or {@code null} if this tree
   * contains no association for the key.
   *
   * <p> A return value of {@code null} does not <i>necessarily</i> indicate that the tree contains
   * no association for the key; it's also possible that the tree explicitly associates the key to
   * {@code null}.
   *
   * @param key the key whose associated value is to be returned
   * @return the value to which the specified key is associated, or {@code null} if this tree
   * contains no association for the key
   */
  V search(K key) {
    return root.getValue(key);
  }

  /**
   * Returns the values associated with the keys specified by the range: {@code key1} and {@code
   * key2}.
   *
   * @param key1 the start key of the range
   * @param policy1 the range policy, {@link RangePolicy#EXCLUSIVE} or {@link
   * RangePolicy#INCLUSIVE}
   * @param key2 the end end of the range
   * @param policy2 the range policy, {@link RangePolicy#EXCLUSIVE} or {@link
   * RangePolicy#INCLUSIVE}
   * @return the values associated with the keys specified by the range: {@code key1} and {@code
   * key2}
   */
  public List<V> searchRange(K key1, RangePolicy policy1, K key2,
      RangePolicy policy2) {
    return root.getRange(key1, policy1, key2, policy2);
  }

  /**
   * Associates the specified value with the specified key in this tree. If the tree previously
   * contained a association for the key, the old value is replaced.
   *
   * @param key the key with which the specified value is to be associated
   * @param value the value to be associated with the specified key
   */
  void insert( K key, V value) {
    this.readLock.lock();

    if(maxKeyInBTree == null || key.compareTo(maxKeyInBTree) > 0) {
      this.chm.insertTuple(key, value);
      this.readLock.unlock();

      if(shouldConvertCHM()) {
        convertCHM();
      }
    }else {
      try {
        this.insertIterative(key, value);
      } catch (Exception e) {
      } finally {
        this.readLock.unlock();
      }
    }
  }

  private void convertCHM() {
    this.writeLock.lock();

    if(!shouldConvertCHM()) {
      this.writeLock.unlock();
      return;
    }

    ImmutablePair<List<K>, List<V>> keysAndValues = this.chm.exportAndClear();
    List<K> keys = keysAndValues.getLeft();
    List<V> values = keysAndValues.getRight();
    this.maxKeyInBTree = keys.get(keys.size() - 1);

    LeafNode node = new LeafNode();

    node.keys = keys;
    node.values = values;

    this.insertLeafNode(node);
    this.writeLock.unlock();

  }

  private void insertLeafNode(LeafNode node) {
    // Special case - root itself is empty;
    if(root.keyNumber() == 0) {
      root = node;
      handleRootOverflow();
      return;
    }

    // B-tree has some data; go to the level above the leaf level
    // It's always true that the first CHM conversion will cause the first leaf node to split
    // and generate an internal node.

    Node iter = root;
    Stack<Node> stack = new Stack<>();

    while(true) {
      if(!iter.isLeaf()){
        stack.push(iter);
        InternalNode temp = (InternalNode) iter;
        iter = temp.children.get(temp.children.size() - 1);
      }else {
        break;
      }
    }

    if(stack.isEmpty()) {
      // only 1 leaf node
      InternalNode newRoot = new InternalNode();

      newRoot.keys.add(node.getFirstLeafKey());
      newRoot.children.add(root);
      newRoot.children.add(node);
      root.next = node;
      root = newRoot;
    }else {
      InternalNode parentAboveLeafLevel = (InternalNode) stack.peek();
      parentAboveLeafLevel.children.get(parentAboveLeafLevel.children.size() - 1).next = node;
      parentAboveLeafLevel.insertChild(node.getFirstLeafKey(), node);

      while(!stack.isEmpty()) {
        rebalance((InternalNode) stack.pop(), stack);
      }
    }
  }

  private void rebalance(InternalNode node, Stack<Node> stack) {
    if(node.isOverflow()) {
      if(stack.isEmpty()) {
        handleRootOverflow();
      }else {
        Node sibling = node.split();
        InternalNode parent = (InternalNode) stack.peek();
        parent.insertChild(sibling.getFirstLeafKey(), sibling);
      }
    }
  }

  private void handleRootOverflow() {
    if (root.isOverflow()) {
      Node sibling = root.split();
      InternalNode newRoot = new InternalNode();
      newRoot.keys.add(sibling.getFirstLeafKey());
      newRoot.children.add(root);
      newRoot.children.add(sibling);
      root = newRoot;
    }
  }

  private boolean shouldConvertCHM() {
    return this.chm.size() > maxCHMSize;
  }

  public void insertIterative(K key, V value) {
    Stack<Node> stack = new Stack<>();
    Node current = root;

    while (current instanceof BPlusTree.InternalNode) {
     current.rl.lock();
      Node rightMost = scannode(key, current);
     current.rl.unlock();
      stack.push(rightMost);
     rightMost.rl.lock();
      current = ((InternalNode) rightMost).getChild(key);
     rightMost.rl.unlock();
    }

    current.wl.lock();
    current = moveRight(key, current);

    if (current instanceof BPlusTree.InternalNode) {
      current.wl.unlock();
      insertIterative(key, value);
    }

    // is safe?
    if (current.keys.size() + 1 == branchingFactor) {
      // not safe
      Node sibling = current.split();

      if (key.compareTo(sibling.keys.get(0)) >= 0) {
        ((LeafNode) sibling).insertValueIterative(key, value);
      } else {
        ((LeafNode) current).insertValueIterative(key, value);
      }

      if (stack.isEmpty()) {
        // root level
        InternalNode newRoot = new InternalNode();
        newRoot.keys.add(sibling.keys.get(0));
        newRoot.children.add(current);
        newRoot.children.add(sibling);

        root = newRoot;
        current.wl.unlock();
      }

      backUp(sibling, stack, current);

    } else {
      ((LeafNode) current).insertValueIterative(key, value);
      current.wl.unlock();
    }
  }

  private void backUp(Node toInsert, Stack<Node> stack, Node oldNode) {
    while (!stack.isEmpty()) {
      Node parent = stack.pop();

      // insert oldnode into parent
      parent.wl.lock();
      Node actual = moveRight(toInsert.keys.get(0), parent);
      oldNode.wl.unlock();

      // is it safe?
      if (actual.keys.size() + 1 == branchingFactor) {
        // not safe
        Node sibling = actual.split();

        if (toInsert.keys.get(0).compareTo(sibling.keys.get(0)) >= 0) {
          ((InternalNode) sibling).insertChild(toInsert.keys.get(0), toInsert);
        } else {
          ((InternalNode) actual).insertChild(toInsert.keys.get(0), toInsert);
        }

        if (stack.isEmpty()) {
          // root
          InternalNode newRoot = new InternalNode();
          newRoot.keys.add(sibling.keys.get(0));
          newRoot.children.add(actual);
          newRoot.children.add(sibling);
          root = newRoot;
          actual.wl.unlock();

        } else {
          toInsert = sibling;
          oldNode = actual;
        }

      } else {
        ((InternalNode) actual).insertChild(toInsert.keys.get(0), toInsert);
        actual.wl.unlock();
        break;
      }
    }
  }

  public Node moveRight(K key, Node current) {
    // todo: either this or compare with current.next.getFirstLeafKey()
    while (current.next != null && key.compareTo(current.next.keys.get(0)) >= 0) {
      current.next.wl.lock();
      current.wl.unlock();
      current = current.next;
    }
    return current;
  }

  private Node scannode(K key, Node current) {
    while (current.next != null && key.compareTo(current.next.keys.get(0)) >= 0) {
      current = current.next;
    }
    return current;
  }


  /**
   * Removes the association for the specified key from this tree if present.
   *
   * @param key the key whose association is to be removed from the tree
   */
  public void delete(K key) {
    root.deleteValue(key);
  }

  public String toString() {
    Queue<List<Node>> queue = new LinkedList<List<Node>>();
    queue.add(Arrays.asList(root));
    StringBuilder sb = new StringBuilder();
    while (!queue.isEmpty()) {
      Queue<List<Node>> nextQueue = new LinkedList<List<Node>>();
      while (!queue.isEmpty()) {
        List<Node> nodes = queue.remove();
        sb.append('{');
        Iterator<Node> it = nodes.iterator();
        while (it.hasNext()) {
          Node node = it.next();
          sb.append(node.toString());
          if (it.hasNext()) {
            sb.append(", ");
          }
          if (node instanceof BPlusTree.InternalNode) {
            nextQueue.add(((InternalNode) node).children);
          }
        }
        sb.append('}');
        if (!queue.isEmpty()) {
          sb.append(", ");
        } else {
          sb.append('\n');
        }
      }
      queue = nextQueue;
    }

    return sb.toString();
  }

  public int numLeaves() {
    Node current = root;
    int count = 0;
    while (current instanceof BPlusTree.InternalNode) {
      current = ((InternalNode) current).children.get(0);
    }
    while (current != null) {
      count += current.keys.size();
      current = current.next;
    }
    return count;

  }

  public static void main(String args[]) {
    // args array: [numThreads, numKeys, branchingFactor]
    int numThreads = Integer.parseInt(args[0]);
    int numKeys = Integer.parseInt(args[1]);
    int branchingFactor = Integer.parseInt(args[2]);
    double maxKeyFactor = Double.parseDouble(args[3]);
    double chmConcurrencyFactor = Double.parseDouble(args[4]);
    int btreeOrChm = Integer.parseInt(args[5]);
    BPlusTree<Integer, Integer> bpTree = new BPlusTree<Integer, Integer>(branchingFactor, (int)Math.ceil(branchingFactor * maxKeyFactor) + numThreads, numThreads, chmConcurrencyFactor);
    ArrayList<Thread> threads = new ArrayList<>();

    long startTime = System.currentTimeMillis();

    for (int i = 0; i < numThreads; i++) {
      Runnable writer = new WriterThread(i, numThreads, numKeys, bpTree, btreeOrChm);
      Thread t = new Thread(writer);
      threads.add(t);
      t.start();
    }
    for (int i = 0; i < numThreads; i++) {
      try {
        threads.get(i).join();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
    }
    long endTime = System.currentTimeMillis();
    System.out.println("Total Time: " + (endTime - startTime));
    // System.out.println(bpTree.numLeaves());
  }

  private abstract class Node {

    ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);
    Lock rl;
    Lock wl;

    Node next;

    List<K> keys;

    int keyNumber() {
      return keys.size();
    }

    abstract V getValue(K key);

    abstract void deleteValue(K key);

    abstract void insertValue(K key, V value);

    abstract K getFirstLeafKey();

    abstract List<V> getRange(K key1, RangePolicy policy1, K key2,
        RangePolicy policy2);

    abstract void merge(Node sibling);

    abstract Node split();

    abstract boolean isOverflow();

    abstract boolean isUnderflow();

    abstract boolean isLeaf();

    public String toString() {
      return keys.toString();
    }
  }

  private class InternalNode extends Node {

    List<Node> children;

    InternalNode() {
      this.keys = new ArrayList<K>();
      this.children = new ArrayList<Node>();
      this.rl = lock.readLock();
      this.wl = lock.writeLock();
    }

    @Override
    V getValue(K key) {
      return getChild(key).getValue(key);
    }

    @Override
    void deleteValue(K key) {
      Node child = getChild(key);
      child.deleteValue(key);
      if (child.isUnderflow()) {
        Node childLeftSibling = getChildLeftSibling(key);
        Node childRightSibling = getChildRightSibling(key);
        Node left = childLeftSibling != null ? childLeftSibling : child;
        Node right = childLeftSibling != null ? child
            : childRightSibling;
        left.merge(right);
        deleteChild(right.getFirstLeafKey());
        if (left.isOverflow()) {
          Node sibling = left.split();
          insertChild(sibling.getFirstLeafKey(), sibling);
        }
        if (root.keyNumber() == 0) {
          root = left;
        }
      }
    }

    @Override
    void insertValue(K key, V value) {
      Node child = getChild(key);
      child.insertValue(key, value);
      if (child.isOverflow()) {
        Node sibling = child.split();
        insertChild(sibling.getFirstLeafKey(), sibling);
      }
      handleRootOverflow();
    }

    @Override
    K getFirstLeafKey() {
      return children.get(0).getFirstLeafKey();
    }

    @Override
    List<V> getRange(K key1, RangePolicy policy1, K key2,
        RangePolicy policy2) {
      return getChild(key1).getRange(key1, policy1, key2, policy2);
    }

    @Override
    void merge(Node sibling) {
      @SuppressWarnings("unchecked")
      InternalNode node = (InternalNode) sibling;
      keys.add(node.getFirstLeafKey());
      keys.addAll(node.keys);
      children.addAll(node.children);

    }

    @Override
    Node split() {
      int from = keyNumber() / 2 + 1, to = keyNumber();
      InternalNode sibling = new InternalNode();
      sibling.keys.addAll(keys.subList(from, to));
      sibling.children.addAll(children.subList(from, to + 1));

      keys.subList(from - 1, to).clear();
      children.subList(from, to + 1).clear();

      sibling.next = this.next;
      this.next = sibling;

      return sibling;
    }

    @Override
    boolean isOverflow() {
      return children.size() > branchingFactor;
    }

    @Override
    boolean isUnderflow() {
      return children.size() < (branchingFactor + 1) / 2;
    }

    @Override
    boolean isLeaf() {
      return false;
    }

    Node getChild(K key) {

      int loc = Collections.binarySearch(keys, key);
      int childIndex = loc >= 0 ? loc + 1 : -loc - 1;
      Node node = children.get(childIndex);
      return node;
    }

    void deleteChild(K key) {
      int loc = Collections.binarySearch(keys, key);
      if (loc >= 0) {
        keys.remove(loc);
        children.remove(loc + 1);
      }
    }

    void insertChild(K key, Node child) {
      int loc = Collections.binarySearch(keys, key);
      int childIndex = loc >= 0 ? loc + 1 : -loc - 1;
      if (loc >= 0) {
        children.set(childIndex, child);
      } else {
        keys.add(childIndex, key);
        children.add(childIndex + 1, child);
      }
    }

    Node getChildLeftSibling(K key) {
      int loc = Collections.binarySearch(keys, key);
      int childIndex = loc >= 0 ? loc + 1 : -loc - 1;
      if (childIndex > 0) {
        return children.get(childIndex - 1);
      }

      return null;
    }

    Node getChildRightSibling(K key) {
      int loc = Collections.binarySearch(keys, key);
      int childIndex = loc >= 0 ? loc + 1 : -loc - 1;
      if (childIndex < keyNumber()) {
        return children.get(childIndex + 1);
      }

      return null;
    }
  }

  private class LeafNode extends Node {

    List<V> values;

    LeafNode() {
      keys = new ArrayList<K>();
      values = new ArrayList<V>();

      this.rl = lock.readLock();
      this.wl = lock.writeLock();
    }

    @Override
    V getValue(K key) {
      int loc = Collections.binarySearch(keys, key);
      return loc >= 0 ? values.get(loc) : null;
    }

    @Override
    void deleteValue(K key) {
      int loc = Collections.binarySearch(keys, key);
      if (loc >= 0) {
        keys.remove(loc);
        values.remove(loc);
      }
    }

    @Override
    void insertValue(K key, V value) {
      int loc = Collections.binarySearch(keys, key);
      int valueIndex = loc >= 0 ? loc : -loc - 1;
      if (loc >= 0) {
        values.set(valueIndex, value);
      } else {
        keys.add(valueIndex, key);
        values.add(valueIndex, value);
      }
      handleRootOverflow();
    }

    void insertValueIterative(K key, V value) {
      int loc = Collections.binarySearch(keys, key);
      int valueIndex = loc >= 0 ? loc : -loc - 1;
      if (loc >= 0) {
        values.set(valueIndex, value);
      } else {
        keys.add(valueIndex, key);
        values.add(valueIndex, value);
      }
    }

    @Override
    K getFirstLeafKey() {
      return keys.get(0);
    }

    @Override
    List<V> getRange(K key1, RangePolicy policy1, K key2,
        RangePolicy policy2) {
      List<V> result = new LinkedList<V>();
      LeafNode node = this;
      while (node != null) {
        Iterator<K> kIt = node.keys.iterator();
        Iterator<V> vIt = node.values.iterator();
        while (kIt.hasNext()) {
          K key = kIt.next();
          V value = vIt.next();
          int cmp1 = key.compareTo(key1);
          int cmp2 = key.compareTo(key2);
          if (((policy1 == RangePolicy.EXCLUSIVE && cmp1 > 0) || (policy1 == RangePolicy.INCLUSIVE
              && cmp1 >= 0))
              && ((policy2 == RangePolicy.EXCLUSIVE && cmp2 < 0) || (
              policy2 == RangePolicy.INCLUSIVE && cmp2 <= 0))) {
            result.add(value);
          } else if ((policy2 == RangePolicy.EXCLUSIVE && cmp2 >= 0)
              || (policy2 == RangePolicy.INCLUSIVE && cmp2 > 0)) {
            return result;
          }
        }
        node = (LeafNode) node.next;
      }
      return result;
    }

    @Override
    void merge(Node sibling) {
      @SuppressWarnings("unchecked")
      LeafNode node = (LeafNode) sibling;
      keys.addAll(node.keys);
      values.addAll(node.values);
      next = node.next;
    }

    @Override
    Node split() {
      LeafNode sibling = new LeafNode();
      int from = (keyNumber() + 1) / 2, to = keyNumber();
      sibling.keys.addAll(keys.subList(from, to));
      sibling.values.addAll(values.subList(from, to));

      keys.subList(from, to).clear();
      values.subList(from, to).clear();

      sibling.next = next;
      next = sibling;
      return sibling;
    }

    @Override
    boolean isOverflow() {
      boolean ret = values.size() > branchingFactor - 1;
      return ret;
    }

    @Override
    boolean isUnderflow() {
      boolean ret = values.size() < branchingFactor / 2;
      return ret;
    }

    @Override
    boolean isLeaf() {
      return true;
    }
  }
}

class ReaderThread implements Runnable {

  // implement different types of workloads here probably
  private final int start;
  private final int factor;
  private final int maxLimit;
  private final BPlusTree<Integer, Integer> bpTree;

  ReaderThread(int start, int factor, int maxLimit, BPlusTree<Integer, Integer> bpTree) {
    this.start = start;
    this.factor = factor;
    this.maxLimit = maxLimit;
    this.bpTree = bpTree;
  }

  public void run() {
    for (int i = start; i < maxLimit; i += factor) {
      int searchValue = bpTree.search(i);
      System.out.println("Value searched " + searchValue + "\n");
    }
  }
}

class WriterThread implements Runnable {

  private final int start;
  private final int factor;
  private final int maxLimit;
  private final BPlusTree<Integer, Integer> bpTree;
  private final int btreeOrChm;

  WriterThread(int start, int factor, int maxLimit, BPlusTree<Integer, Integer> bpTree, int btreeOrChm) {
    this.start = start;
    this.factor = factor;
    this.maxLimit = maxLimit;
    this.bpTree = bpTree;
    this.btreeOrChm = btreeOrChm;
  }

  public void run() {
    for (int i = start; i < maxLimit; i += factor) {

      try {
        if (btreeOrChm == 0) { // B-tree insert
          bpTree.insertIterative(i, i);
        } else { // CHM Insert
          bpTree.insert(i, i);
        }
      }catch (Exception e) {
      }
    }
  }
}
