<template>
  <div id="app" data="data">
    <b-container fluid>
      <b-row>
        <b-col cols="7">
          <ShipMap :ships="$store.state.ships" :uboot_pos="$store.state.uboot_pos"/>
        </b-col>
        <b-col cols="2">
          <UBootInput/>
          <TimeControl/>
        </b-col>
        <b-col>
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

export default {
  name: 'App',
  components: {
    TimeControl,
    ShipList,
    ShipMap,
    UBootInput
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
