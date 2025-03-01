"use client"

import * as React from "react"
import { Area, AreaChart, CartesianGrid, XAxis } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

// import {
//   Select,
//   SelectContent,
//   SelectItem,
//   SelectTrigger,
//   SelectValue,
// } from "@/components/ui/select"

const chartData = [
  {
    date: "2024-04-01",
    anger: 222,
    disgust: 150,
    fear: 189,
    happy: 321,
    neutral: 178,
    pleasant_surprised: 230,
    sad: 99,
  },
  {
    date: "2024-04-02",
    anger: 97,
    disgust: 180,
    fear: 150,
    happy: 245,
    neutral: 98,
    pleasant_surprised: 120,
    sad: 220,
  },
  {
    date: "2024-04-03",
    anger: 167,
    disgust: 120,
    fear: 110,
    happy: 350,
    neutral: 98,
    pleasant_surprised: 310,
    sad: 140,
  },
  {
    date: "2024-04-04",
    anger: 242,
    disgust: 260,
    fear: 220,
    happy: 400,
    neutral: 165,
    pleasant_surprised: 360,
    sad: 200,
  },
  {
    date: "2024-04-05",
    anger: 373,
    disgust: 290,
    fear: 210,
    happy: 500,
    neutral: 200,
    pleasant_surprised: 480,
    sad: 100,
  },
  {
    date: "2024-04-06",
    anger: 301,
    disgust: 340,
    fear: 160,
    happy: 420,
    neutral: 170,
    pleasant_surprised: 380,
    sad: 130,
  },
  {
    date: "2024-04-07",
    anger: 245,
    disgust: 180,
    fear: 200,
    happy: 380,
    neutral: 150,
    pleasant_surprised: 290,
    sad: 160,
  },
  {
    date: "2024-04-08",
    anger: 409,
    disgust: 320,
    fear: 330,
    happy: 550,
    neutral: 210,
    pleasant_surprised: 460,
    sad: 190,
  },
  {
    date: "2024-04-09",
    anger: 59,
    disgust: 110,
    fear: 130,
    happy: 180,
    neutral: 50,
    pleasant_surprised: 90,
    sad: 30,
  },
  {
    date: "2024-04-10",
    anger: 261,
    disgust: 190,
    fear: 210,
    happy: 320,
    neutral: 100,
    pleasant_surprised: 270,
    sad: 120,
  },
  {
    date: "2024-04-11",
    anger: 327,
    disgust: 350,
    fear: 270,
    happy: 450,
    neutral: 220,
    pleasant_surprised: 400,
    sad: 180,
  },
  {
    date: "2024-04-12",
    anger: 292,
    disgust: 210,
    fear: 220,
    happy: 350,
    neutral: 120,
    pleasant_surprised: 300,
    sad: 160,
  },
  {
    date: "2024-04-13",
    anger: 342,
    disgust: 380,
    fear: 310,
    happy: 470,
    neutral: 210,
    pleasant_surprised: 440,
    sad: 130,
  },
  {
    date: "2024-04-14",
    anger: 137,
    disgust: 220,
    fear: 180,
    happy: 290,
    neutral: 100,
    pleasant_surprised: 250,
    sad: 190,
  },
  {
    date: "2024-04-15",
    anger: 120,
    disgust: 170,
    fear: 160,
    happy: 260,
    neutral: 110,
    pleasant_surprised: 220,
    sad: 110,
  },
  {
    date: "2024-04-16",
    anger: 138,
    disgust: 190,
    fear: 130,
    happy: 310,
    neutral: 95,
    pleasant_surprised: 230,
    sad: 140,
  },
  {
    date: "2024-04-17",
    anger: 446,
    disgust: 360,
    fear: 280,
    happy: 530,
    neutral: 240,
    pleasant_surprised: 460,
    sad: 130,
  },
  {
    date: "2024-04-18",
    anger: 364,
    disgust: 410,
    fear: 380,
    happy: 490,
    neutral: 230,
    pleasant_surprised: 500,
    sad: 160,
  },
  {
    date: "2024-04-19",
    anger: 243,
    disgust: 180,
    fear: 200,
    happy: 360,
    neutral: 140,
    pleasant_surprised: 270,
    sad: 180,
  },
  {
    date: "2024-04-20",
    anger: 89,
    disgust: 150,
    fear: 100,
    happy: 220,
    neutral: 130,
    pleasant_surprised: 180,
    sad: 50,
  },
  {
    date: "2024-04-21",
    anger: 137,
    disgust: 200,
    fear: 180,
    happy: 270,
    neutral: 160,
    pleasant_surprised: 230,
    sad: 140,
  },
  {
    date: "2024-04-22",
    anger: 224,
    disgust: 170,
    fear: 210,
    happy: 360,
    neutral: 180,
    pleasant_surprised: 300,
    sad: 130,
  },
  {
    date: "2024-04-23",
    anger: 138,
    disgust: 230,
    fear: 190,
    happy: 310,
    neutral: 150,
    pleasant_surprised: 250,
    sad: 160,
  },
  {
    date: "2024-04-24",
    anger: 387,
    disgust: 290,
    fear: 280,
    happy: 470,
    neutral: 200,
    pleasant_surprised: 380,
    sad: 130,
  },
  {
    date: "2024-04-25",
    anger: 215,
    disgust: 250,
    fear: 230,
    happy: 420,
    neutral: 170,
    pleasant_surprised: 330,
    sad: 120,
  },
  {
    date: "2024-04-26",
    anger: 75,
    disgust: 130,
    fear: 80,
    happy: 160,
    neutral: 50,
    pleasant_surprised: 110,
    sad: 30,
  },
  {
    date: "2024-04-27",
    anger: 383,
    disgust: 420,
    fear: 330,
    happy: 530,
    neutral: 210,
    pleasant_surprised: 490,
    sad: 180,
  },
  {
    date: "2024-04-28",
    anger: 122,
    disgust: 180,
    fear: 200,
    happy: 330,
    neutral: 90,
    pleasant_surprised: 250,
    sad: 120,
  },
  {
    date: "2024-04-29",
    anger: 315,
    disgust: 240,
    fear: 210,
    happy: 380,
    neutral: 180,
    pleasant_surprised: 290,
    sad: 150,
  },
  {
    date: "2024-04-30",
    anger: 454,
    disgust: 380,
    fear: 310,
    happy: 550,
    neutral: 230,
    pleasant_surprised: 500,
    sad: 180,
  },
  // More data can be added similarly
]

const chartConfig = {
  anger: {
    label: "Anger",
    color: "#ef4444", // Red
  },
  disgust: {
    label: "Disgust",
    color: "#84cc16", // Lime
  },
  fear: {
    label: "Fear",
    color: "#8b5cf6", // Purple
  },
  happy: {
    label: "Happy",
    color: "#f59e0b", // Amber
  },
  neutral: {
    label: "Neutral",
    color: "#94a3b8", // Slate
  },
  pleasant_surprised: {
    label: "Pleasant Surprise",
    color: "#06b6d4", // Cyan
  },
  sad: {
    label: "Sad",
    color: "#4f46e5", // Indigo
  },
}

export function MainGraph() {
  const [timeRange, setTimeRange] = React.useState("90d")

  // Define these colors for use in gradients
  const desktopColor = "#4f46e5" // Indigo
  const mobileColor = "#06b6d4" // Cyan

  const filteredData = chartData.filter((item) => {
    const date = new Date(item.date)
    const referenceDate = new Date("2024-06-30")
    let daysToSubtract = 90
    if (timeRange === "30d") {
      daysToSubtract = 30
    } else if (timeRange === "7d") {
      daysToSubtract = 7
    }
    const startDate = new Date(referenceDate)
    startDate.setDate(startDate.getDate() - daysToSubtract)
    return date >= startDate
  })

  return (
    <Card>
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>Mood Trends</CardTitle>
          <CardDescription>Tracking emotional well-being</CardDescription>
        </div>
        {/* <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger
            className="w-[160px] rounded-lg sm:ml-auto"
            aria-label="Select a value"
          >
            <SelectValue placeholder="Last 3 months" />
          </SelectTrigger>
          <SelectContent className="rounded-xl">
            <SelectItem value="90d" className="rounded-lg">
              Last 3 months
            </SelectItem>
            <SelectItem value="30d" className="rounded-lg">
              Last 30 days
            </SelectItem>
            <SelectItem value="7d" className="rounded-lg">
              Last 7 days
            </SelectItem>
          </SelectContent>
        </Select> */}
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[250px] w-full"
        >
          <AreaChart data={filteredData}>
            <defs>
              {Object.entries(chartConfig).map(([key, config]) => (
                <linearGradient
                  key={`fill-${key}`}
                  id={`fill-${key}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="5%"
                    stopColor={config.color}
                    stopOpacity={0.8}
                  />
                  <stop
                    offset="95%"
                    stopColor={config.color}
                    stopOpacity={0.1}
                  />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={(value) => {
                const date = new Date(value)
                return date.toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })
              }}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => {
                    return new Date(value).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    })
                  }}
                  indicator="dot"
                />
              }
            />
            {Object.entries(chartConfig).map(([key, config]) => (
              <Area
                key={key}
                dataKey={key}
                type="monotone"
                fill={`url(#fill-${key})`}
                stroke={config.color}
                strokeWidth={2}
              />
            ))}
            <ChartLegend content={<ChartLegendContent />} />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
