<template>
  <div>
    <div v-for="ship in ships" :key="ship.id">
      <ShipCard
          :ship="ship"
          :id="ship.id"
      />
    </div>
    <AddShipForm @add-ship="(ship_data) => $store.commit('add_ship', ship_data)"/>
  </div>
</template>

<script>
import AddShipForm from "@/components/AddShipForm";
import ShipCard from "@/components/ShipCard";
import {mapState} from 'vuex'

export default {
  name: "ShipLIst",
  components: {AddShipForm, ShipCard},
  data() {
    return {
      timer: null,
    }
  },
  computed: mapState(['ships']),
  created() {
    this.$store.commit('add_ship', {name: "Boaty", tons: 1000, ship_type: "Unknown", speed: 10.0})
  },

}
</script>

<style scoped>

</style>