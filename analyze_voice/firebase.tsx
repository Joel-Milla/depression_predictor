// Import the functions you need from the SDKs you need

import { getAnalytics } from "firebase/analytics"
import { initializeApp } from "firebase/app"
import { collection, getDocs, getFirestore } from "firebase/firestore"

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
    const users = []
    const usersCollection = collection(db, "users")
    let querySnapshot = await getDocs(usersCollection)

    querySnapshot.forEach((doc) => {
      // Add document data and id to array
      users.push({
        id: doc.id,
        ...doc.data(),
      })
    })

    const videos = []
    const videosCollection = collection(db, "videos")
    querySnapshot = await getDocs(videosCollection)

    querySnapshot.forEach((doc) => {
      // Add document data and id to array
      videos.push({
        id: doc.id,
        ...doc.data(),
      })
    })

    const results = []
    const resultsCollection = collection(db, "videos")
    querySnapshot = await getDocs(resultsCollection)

    querySnapshot.forEach((doc) => {
      // Add document data and id to array
      results.push({
        id: doc.id,
        ...doc.data(),
      })
    })

    return users
  } catch (error) {
    console.error("Error fetching users:", error)
    throw error
  }
}

// Usage example
// You can call this function from anywhere in your application
// For example, in a React component:

/*
useEffect(() => {
  async function loadUsers() {
    const usersData = await fetchAllUsers()
    setUsers(usersData) // Assuming you have a state variable for users
  }
  
  loadUsers()
}, [])
*/

export { fetchAllUsers }
