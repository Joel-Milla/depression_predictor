"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { initializeApp } from "firebase/app"
import {
  collection,
  doc,
  getDoc,
  getDocs,
  getFirestore,
  query,
  where,
} from "firebase/firestore"

import { EmotionsBarChart } from "@/components/General/EmotionsBarChart"
import { EmotionsPieChart } from "@/components/General/EmotionsPieChart"

// Firebase config
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
const db = getFirestore(app)

export default function IndexPage() {
  const params = useParams()
  const phoneId = params.id // Get the phone ID from URL

  const [emotionData, setEmotionData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true)

        // Direct fetch for the specific user by phone ID
        const userDoc = await getDoc(doc(db, "users", phoneId))

        if (!userDoc.exists()) {
          console.error("User not found")
          setEmotionData(fallbackEmotionData)
          return
        }

        const userData = {
          id: userDoc.id,
          ...userDoc.data(),
          phone: userDoc.id,
        }

        // Get videos for this user
        const videosQuery = query(
          collection(db, "videos"),
          where("recording_phone", "==", phoneId)
        )
        const videosSnapshot = await getDocs(videosQuery)
        const userVideos = videosSnapshot.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }))

        // Get results for these videos
        const videoIds = userVideos.map((video) => video.recording_sid)
        const resultsSnapshot = await getDocs(collection(db, "results"))
        const resultsData = resultsSnapshot.docs
          .map((doc) => ({
            id: doc.id,
            ...doc.data(),
          }))
          .filter((result) => videoIds.includes(result.id))

        // Combine videos with their results
        const videosWithResults = userVideos.map((video) => {
          const videoResults = resultsData.find(
            (result) => result.id === video.recording_sid
          )

          return {
            ...video,
            results: videoResults || null,
          }
        })

        // Process the data
        const userData2 = {
          ...userData,
          videos: videosWithResults,
        }

        const processedData = processEmotionData([userData2])
        setEmotionData(processedData)
      } catch (error) {
        console.error("Error fetching emotion data:", error)
        setEmotionData(fallbackEmotionData)
      } finally {
        setLoading(false)
      }
    }

    if (phoneId) {
      fetchData()
    } else {
      setEmotionData(fallbackEmotionData)
      setLoading(false)
    }
  }, [phoneId])

  // Process data for charts
  function processEmotionData(usersData) {
    const emotions = {
      anger: 0,
      disgust: 0,
      fear: 0,
      happy: 0,
      neutral: 0,
      pleasant_surprised: 0,
      sad: 0,
    }

    usersData.forEach((user) => {
      if (user.videos) {
        user.videos.forEach((video) => {
          if (video.results && video.results.emotions) {
            Object.keys(emotions).forEach((emotion) => {
              emotions[emotion] += video.results.emotions[emotion] || 0
            })
          }
        })
      }
    })

    return [
      { emotion: "anger", count: emotions.anger, fill: "#ef4444" },
      { emotion: "disgust", count: emotions.disgust, fill: "#84cc16" },
      { emotion: "fear", count: emotions.fear, fill: "#8b5cf6" },
      { emotion: "happy", count: emotions.happy, fill: "#f59e0b" },
      { emotion: "neutral", count: emotions.neutral, fill: "#94a3b8" },
      {
        emotion: "pleasant_surprised",
        count: emotions.pleasant_surprised,
        fill: "#06b6d4",
      },
      { emotion: "sad", count: emotions.sad, fill: "#4f46e5" },
    ]
  }

  // Fallback data
  const fallbackEmotionData = [
    { emotion: "anger", count: 99, fill: "#ef4444" },
    { emotion: "disgust", count: 321, fill: "#84cc16" },
    { emotion: "fear", count: 222, fill: "#8b5cf6" },
    { emotion: "happy", count: 102, fill: "#f59e0b" },
    { emotion: "neutral", count: 178, fill: "#94a3b8" },
    { emotion: "pleasant_surprised", count: 230, fill: "#06b6d4" },
    { emotion: "sad", count: 150, fill: "#4f46e5" },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
      {loading ? (
        <>
          <div className="w-full h-72 bg-gray-100 animate-pulse rounded-lg"></div>
          <div className="w-full h-72 bg-gray-100 animate-pulse rounded-lg"></div>
        </>
      ) : (
        <>
          <EmotionsPieChart data={emotionData} />
          <EmotionsBarChart data={emotionData} />
        </>
      )}
    </div>
  )
}
