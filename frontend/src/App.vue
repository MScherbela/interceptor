<template>
  <div id="app" data="data">
    <b-container fluid>
      <b-row>
        <b-col>
          <ShipMap :ships="$store.state.ships" :uboot_pos="$store.state.uboot_pos" :intercept="$store.state.intercept"/>
        </b-col>
        <b-col cols="3">
          <UBootInput/>
          <TimeControl/>
          <InterceptControl
              :ships="$store.state.ships"
              :uboot_pos="$store.state.uboot_pos"
              :time="this.$store.state.time"
              :intercept="$store.state.intercept"
          />
        </b-col>
        <b-col cols="3">
          <ShipList/>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import ShipList from './components/ShipLIst.vue'
import ShipMap from "./components/ShipMap";
import UBootInput from "@/components/UBootInput";
import TimeControl from "@/components/TimeControl";
import InterceptControl from "@/components/InterceptControl";

export default {
  name: 'App',
  components: {
    TimeControl,
    ShipList,
    ShipMap,
    UBootInput,
    InterceptControl
  },

}
</script>

<style>
/*#app {*/
/*  font-family: Avenir, Helvetica, Arial, sans-serif;*/
/*  -webkit-font-smoothing: antialiased;*/
/*  -moz-osx-font-smoothing: grayscale;*/
/*  text-align: center;*/
/*  color: #2c3e50;*/
/*  margin-top: 60px;*/
/*}*/
</style>
