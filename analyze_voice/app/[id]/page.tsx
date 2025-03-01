import { EmotionsBarChart } from "@/components/General/EmotionsBarChart"
import { EmotionsPieChart } from "@/components/General/EmotionsPieChart"

export default function IndexPage() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
      <EmotionsPieChart />
      <EmotionsBarChart />
    </div>
  )
}
