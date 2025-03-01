"use client"

import { TrendingUp } from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

// Emotion data for bar chart
// const chartData = [
//   { emotion: "Anger", count: 222 },
//   { emotion: "Disgust", count: 150 },
//   { emotion: "Fear", count: 189 },
//   { emotion: "Happy", count: 321 },
//   { emotion: "Neutral", count: 178 },
//   { emotion: "Pleasant", count: 230 },
//   { emotion: "Sad", count: 99 },
// ]

// Chart configuration with colors for emotions
const chartConfig = {
  count: {
    label: "Count",
    color: "#4f46e5", // Default color
  },
  Anger: {
    label: "Anger",
    color: "#ef4444", // Red
  },
  Disgust: {
    label: "Disgust",
    color: "#84cc16", // Lime
  },
  Fear: {
    label: "Fear",
    color: "#8b5cf6", // Purple
  },
  Happy: {
    label: "Happy",
    color: "#f59e0b", // Amber
  },
  Neutral: {
    label: "Neutral",
    color: "#94a3b8", // Slate
  },
  Pleasant: {
    label: "Pleasant",
    color: "#06b6d4", // Cyan
  },
  Sad: {
    label: "Sad",
    color: "#4f46e5", // Indigo
  },
}

export function EmotionsBarChart({ data }) {
  // Calculate the most prevalent emotion
  const mostPrevalent = [...data].sort((a, b) => b.count - a.count)[0]
  const totalEmotions = data.reduce((sum, item) => sum + item.count, 0)
  const prevalentPercentage = (
    (mostPrevalent.count / totalEmotions) *
    100
  ).toFixed(1)
  return (
    <Card className="w-full">
      <CardHeader className="pb-2">
        <CardTitle>Emotion Analysis</CardTitle>
        <CardDescription>Distribution of detected emotions</CardDescription>
      </CardHeader>
      <CardContent className="p-4">
        <ChartContainer config={chartConfig} className="h-100">
          {" "}
          {/* Increase height to match pie chart */}
          <BarChart data={data}>
            <CartesianGrid
              vertical={false}
              strokeDasharray="3 3"
              opacity={0.3}
            />
            <XAxis
              dataKey="emotion"
              tickLine={false}
              tickMargin={8}
              axisLine={false}
              fontSize={12}
              tickFormatter={(value) => value.slice(0, 3)} // Abbreviate to first 3 letters
            />
            <YAxis tickLine={false} axisLine={false} fontSize={12} />
            <ChartTooltip
              cursor={{ fill: "rgba(0, 0, 0, 0.05)" }}
              content={<ChartTooltipContent />}
            />
            <Bar
              dataKey="count"
              radius={[4, 4, 0, 0]}
              barSize={30}
              // Apply different colors to each bar
              fill="#4f46e5"
              className="emotion-bars"
            />
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-1 text-sm pt-0">
        <div className="flex gap-2 font-medium leading-none">
          {mostPrevalent.emotion} is the dominant emotion at{" "}
          {prevalentPercentage}% <TrendingUp className="h-4 w-4" />
        </div>
        <div className="leading-none text-muted-foreground">
          Based on analysis of {totalEmotions} emotional signals
        </div>
      </CardFooter>
    </Card>
  )
}
