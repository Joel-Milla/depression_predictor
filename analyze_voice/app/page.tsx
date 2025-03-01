"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { fetchAllUsers } from "@/firebase"

import { Skeleton } from "@/components/ui/skeleton"
import { MainGraph } from "@/components/General/MainGraph"
// import { siteConfig } from "@/config/site"
// import { buttonVariants } from "@/components/ui/button"
import { DataTable } from "@/components/table/table"

const data: Payment[] = [
  {
    id: "m5gr84i9",
    phone: "316-555-0123",
    status: "success",
    date: new Date("December 17, 1995 03:24:00"),
    name: "Ken Adams",
  },
  {
    id: "3u1reuv4",
    phone: "242-555-0198",
    status: "success",
    date: new Date("March 5, 2001 10:15:30"),
    name: "Abe Lincoln",
  },
  {
    id: "derv1ws0",
    phone: "837-555-0147",
    status: "pending",
    date: new Date("June 12, 2015 14:45:00"),
    name: "Monserrat Rivera",
  },
  {
    id: "5kma53ae",
    phone: "874-555-0162",
    status: "success",
    date: new Date("August 21, 2018 08:30:00"),
    name: "Silas Thompson",
  },
  {
    id: "bhqecj4p",
    phone: "721-555-0173",
    status: "failed",
    date: new Date("January 2, 2020 19:20:45"),
    name: "Carmella Johnson",
  },
  {
    id: "y3er8ks2",
    phone: "654-555-0135",
    status: "success",
    date: new Date("November 10, 2013 22:10:00"),
    name: "James Carter",
  },
  {
    id: "p9dm4xw7",
    phone: "235-555-0189",
    status: "success",
    date: new Date("July 18, 2016 12:55:00"),
    name: "Lena Clarkson",
  },
  {
    id: "v8sl3ek0",
    phone: "458-555-0124",
    status: "failed",
    date: new Date("September 29, 2022 06:40:00"),
    name: "Daniel Foster",
  },
  {
    id: "u7kc2eq5",
    phone: "369-555-0158",
    status: "success",
    date: new Date("April 14, 2005 17:25:00"),
    name: "Sophia Martinez",
  },
  {
    id: "t2we9dj3",
    phone: "589-555-0179",
    status: "failed",
    date: new Date("May 30, 2011 20:05:00"),
    name: "Olivia Chang",
  },
  {
    id: "a5xz1mw4",
    phone: "902-555-0194",
    status: "failed",
    date: new Date("February 25, 2017 09:10:00"),
    name: "Ethan Wright",
  },
  {
    id: "w6cn9xa8",
    phone: "103-555-0183",
    status: "success",
    date: new Date("October 8, 2019 15:45:00"),
    name: "Mia Anderson",
  },
  {
    id: "b4yx7qp9",
    phone: "741-555-0119",
    status: "pending",
    date: new Date("March 13, 2023 11:30:00"),
    name: "Noah Bennett",
  },
  {
    id: "z9cm5ep2",
    phone: "825-555-0120",
    status: "success",
    date: new Date("December 5, 2008 04:50:00"),
    name: "Emma Lee",
  },
  {
    id: "d2kl8sy3",
    phone: "369-555-0185",
    status: "failed",
    date: new Date("July 7, 2021 13:20:00"),
    name: "Liam Rodriguez",
  },
]
export type Payment = {
  id: string
  phone: string
  date: Date
  status: "pending" | "success" | "failed"
  name: string
  metadata?: Record<string, any>
}

export default function IndexPage() {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadUsers() {
      const usersData = await fetchAllUsers()
      console.log(usersData)
      const combinedData = [...usersData, ...data]
      setUsers(combinedData) // Assuming you have a state variable for users
      setLoading(false)
    }

    loadUsers()
  }, [])

  return (
    <div className="m-5">
      {loading ? (
        // Simple skeleton placeholders
        <div>
          <div className="space-y-2 mb-8">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-60 w-full" />
          </div>
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-60 w-full" />
          </div>
        </div>
      ) : (
        <>
          <DataTable data={users} />
          <MainGraph />
        </>
      )}
    </div>
  )
  return (
    <></>
    // <section className="container grid items-center gap-6 pb-8 pt-6 md:py-10">
    //   <div className="flex max-w-[980px] flex-col items-start gap-2">
    //     <h1 className="text-3xl font-extrabold leading-tight tracking-tighter md:text-4xl">
    //       Beautifully designed components <br className="hidden sm:inline" />
    //       built with Radix UI and Tailwind CSS.
    //     </h1>
    //     <p className="max-w-[700px] text-lg text-muted-foreground">
    //       Accessible and customizable components that you can copy and paste
    //       into your apps. Free. Open Source. And Next.js 13 Ready.
    //     </p>
    //   </div>
    //   <div className="flex gap-4">
    //     <Link
    //       href={siteConfig.links.docs}
    //       target="_blank"
    //       rel="noreferrer"
    //       className={buttonVariants()}
    //     >
    //       Documentation
    //     </Link>
    //     <Link
    //       target="_blank"
    //       rel="noreferrer"
    //       href={siteConfig.links.github}
    //       className={buttonVariants({ variant: "outline" })}
    //     >
    //       GitHub
    //     </Link>
    //   </div>
    // </section>
  )
}
