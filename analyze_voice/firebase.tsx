// Import the functions you need from the SDKs you need

import { getAnalytics } from "firebase/analytics"
import { initializeApp } from "firebase/app"
import {
  collection,
  doc,
  getDoc,
  getDocs,
  getFirestore,
} from "firebase/firestore"

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAZu9jEtGQr7BQiN23ZrrlBToNB4bVrqT8",
  authDomain: "ai-caller-9c525.firebaseapp.com",
  projectId: "ai-caller-9c525",
  storageBucket: "ai-caller-9c525.firebasestorage.app",
  messagingSenderId: "977456309191",
  appId: "1:977456309191:web:516c9ad4fe4ce75ec46fe2",
  measurementId: "G-TVJ5WB5164",
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)
const analytics = getAnalytics(app)

// Initialize Firestore
const db = getFirestore(app)

// Function to fetch all users
async function fetchAllUsers() {
  try {
    const TARGET_PHONES = ["+6591268057", "+6582924032"]

    const payments = []

    const usersCollection = collection(db, "users")
    const videosCollection = collection(db, "videos")
    const resultsCollection = collection(db, "results")

    const [usersSnapshot, videosSnapshot, resultsSnapshot] = await Promise.all([
      getDocs(usersCollection),
      getDocs(videosCollection),
      getDocs(resultsCollection),
    ])

    const users = usersSnapshot.docs.map((doc) => ({
      id: doc.id, // This is the phone number
      ...doc.data(),
      phone: doc.id,
    }))

    const videos = videosSnapshot.docs.map((doc) => ({
      id: doc.id, // This is the call_sid
      ...doc.data(),
    }))

    const results = resultsSnapshot.docs.map((doc) => ({
      id: doc.id, // This is the recording ID
      ...doc.data(),
    }))

    // Step 2: Process each target phone number
    for (const targetPhone of TARGET_PHONES) {
      // Find the user with this phone number
      const user = users.find((u) => u.phone === targetPhone)

      if (user) {
        // Find videos associated with this user
        const userVideos = videos.filter(
          (video) => video.recording_phone === targetPhone
        )

        // For each video, find and attach results
        const videosWithResults = userVideos.map((video) => {
          const videoResults = results.find(
            (result) => result.id === video.call_sid
          )

          return {
            ...video,
            results: videoResults || null,
          }
        })

        // Create a Payment object for this user
        const payment = {
          id: user.id,
          phone: targetPhone,
          name: user.name || user.display_name || "Unknown User",
          // Use date from first video if available, otherwise current date
          date:
            userVideos.length > 0 && userVideos[0].recording_date
              ? new Date(userVideos[0].recording_date)
              : new Date(),
          // Status based on whether they have videos
          status: userVideos.length > 0 ? "success" : "pending",
          // Include all user data and videos in metadata
          metadata: {
            userData: user,
            videos: videosWithResults,
          },
        }

        payments.push(payment)
      } else {
        // If user not found, create a minimal Payment object
        payments.push({
          id: targetPhone,
          phone: targetPhone,
          name: "Unknown User",
          date: new Date(),
          status: "pending",
          metadata: {},
        })
      }
    }

    return payments
  } catch (error) {
    console.error("Error fetching and combining data:", error)
    throw error
  }
}

export { fetchAllUsers }
