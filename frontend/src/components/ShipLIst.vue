<template>
  <div>
    <div v-for="ship in ships" :key="ship.id">
      <ShipCard
          :ship="ship"
          :id="ship.id"
          class="mt-0 mb-2"
      />
    </div>
    <b-card class="cardbackground" body-class="compactcardbody">
      <AddShipForm/>
    </b-card>
  </div>
</template>

<script>
import AddShipForm from "@/components/AddShipForm";
import ShipCard from "@/components/ShipCard";
import {mapState} from 'vuex'

export default {
  name: "ShipLIst",
  components: {AddShipForm, ShipCard},
  computed: mapState(['ships']),
}
</script>

<style scoped>

</style>