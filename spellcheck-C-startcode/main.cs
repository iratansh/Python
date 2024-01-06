// Spell Check Starter
// This start code creates two lists
// 1: dictionary: an array containing all of the words from "dictionary.txt"
// 2: aliceWords: an array containing all of the words from "AliceInWonderland.txt"

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.RegularExpressions;

class Program {
  public static void Main (string[] args) {
    // Load data files into arrays
    String[] dictionary = System.IO.File.ReadAllLines(@"data-files/dictionary.txt");
    String aliceText = System.IO.File.ReadAllText(@"data-files/AliceInWonderLand.txt");
    String[] aliceWords = Regex.Split(aliceText, @"\s+");
    
    List<string> dictionaryList = new List<string>(dictionary);
    List<string> aliceWordsList = new List<string>(aliceWords);
    
    while (true) {
            int selection = getMenuSelection();

            switch (selection) {
                case 1:
                    Console.WriteLine("Type a word to Search: ");
                    string input1 = Console.ReadLine().ToLower();
                    Stopwatch stopwatch1 = new Stopwatch();
                    stopwatch1.Start();
                    int search1 = linearSearch(dictionaryList, input1);
                    stopwatch1.Stop();
                    TimeSpan ts1 = stopwatch1.Elapsed;
                    if (search1 == -1) {
                      Console.WriteLine("Word Not Found");
                    } else {
                      Console.WriteLine("Word found at Index " + search1 + " Time Elapsed: " + ts1);
                    }
                  
                    break;
                case 2:
                    Console.WriteLine("Type a word to Search: ");
                    string input2 = Console.ReadLine().ToLower();
                    Stopwatch stopwatch2 = new Stopwatch();
                    stopwatch2.Start();
                    int search2 = binarySearch(dictionaryList, input2);
                    stopwatch2.Stop();
                    TimeSpan ts2 = stopwatch2.Elapsed;
                    if (search2 == -1) {
                        Console.WriteLine("Word Not Found");
                    } else {
                        Console.WriteLine("Word found at Index " + search2 + " Time Elapsed: " + ts2);
                    }
              
                    break;
                case 3:
                    Stopwatch stopwatch3 = new Stopwatch();
                    int count1 = 0;
                    
                    stopwatch3.Start();
                    for (int i = 0; i < aliceWordsList.Count; i++) {
                      int search3 = linearSearch(dictionaryList, aliceWordsList[i]);
                      if (search3 == -1) {
                        count1 += 1;
                      }
                    }
                    stopwatch3.Stop();
                    TimeSpan ts3 = stopwatch3.Elapsed;
                    Console.WriteLine("Words Found in Dictionary: " + count1 + " Time Elapsed: " + ts3);

                    stopwatch3.Reset();
                    break;
                case 4:
                    Stopwatch stopwatch4 = new Stopwatch();
                    int count2 = 0;
                    
                    stopwatch4.Start();
                    for (int i = 0; i < aliceWordsList.Count; i++) {
                      int search4 = binarySearch(dictionaryList, aliceWordsList[i]);
                      if (search4 == -1) {
                        count2 += 1;
                      }
                    }
                    stopwatch4.Stop();
                    TimeSpan ts4 = stopwatch4.Elapsed;
                    Console.WriteLine("Words Found in Dictionary: " + count2 + " Time Elapsed: " + ts4);

                    break;
                case 5:
                    return;
                default:
                    Console.WriteLine("Invalid selection. Please choose a number between 1 and 5.");
                    break;
            }

    }
  }
  
  static int getMenuSelection() {
    Console.WriteLine("\nMY CONTACTS MENU");
    Console.WriteLine("1: Spell Check a Word (Linear Search)");
    Console.WriteLine("2: Spell Check a Word (Binary Search)");
    Console.WriteLine("3: Alice in Wonderland (Linear Search)");
    Console.WriteLine("4: Alice in Wonderland (Binary Search)");
    Console.WriteLine("5: Exit");
    Console.Write("Selection (1-5): ");
    
    int selection;
    while (!int.TryParse(Console.ReadLine(), out selection) || selection < 1 || selection > 5) {
            Console.WriteLine("Invalid input. Please enter a valid number between 1 and 5.");
            Console.Write("Selection (1-7): ");
        }

      return selection;
    }

  static int linearSearch<T>(List<T> list, T item) {
    for (int i = 0; i < list.Count; i++) {
      if (EqualityComparer<T>.Default.Equals(list[i], item)) {
        return i;
      }
    }
    return -1;
  }
    
  static int binarySearch<T>(List<T> list, T item) {
    int lowerIndex = 0;
    int upperIndex = list.Count - 1;
  
    while (lowerIndex <= upperIndex) {
      int middleIndex = (lowerIndex + upperIndex) / 2;
      int comparisonResult = Comparer<T>.Default.Compare(item, list[middleIndex]);
  
      if (comparisonResult == 0) {
        return middleIndex;
      } else if (comparisonResult < 0) {
        upperIndex = middleIndex - 1;
      } else {
        lowerIndex = middleIndex + 1;
      }
    }
    return -1;
  }

}